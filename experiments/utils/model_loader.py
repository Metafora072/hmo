"""
HMO Research — Model Loader
Handles loading Qwen3.5 hybrid models (BF16 / GPTQ-Int4) with proper device mapping.
Model root defaults to the local SSD path, but can be overridden with
`HMO_MODEL_ROOT` for rented-server data-disk deployments.
"""
import os
import torch
from pathlib import Path
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def _fix_gptqmodel_triton_patch():
    """Fix gptqmodel's broken triton Autotuner patch (missing _cache_lock).
    Instead of reverting the patch, add the missing attribute."""
    try:
        import threading
        import gptqmodel.utils.nogil_patcher  # noqa: F401
        import triton
        autotuner_cls = triton.runtime.autotuner.Autotuner
        # Add missing _cache_lock if the patch expects it
        orig_init = autotuner_cls.__init__
        def _patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if not hasattr(self, '_cache_lock'):
                self._cache_lock = threading.Lock()
        autotuner_cls.__init__ = _patched_init
    except Exception:
        pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(os.environ.get("HMO_MODEL_ROOT", str(PROJECT_ROOT / "models"))).expanduser()


def _build_multi_gpu_max_memory(
    reserve_gib: float = 8.0,
    first_gpu_extra_reserve_gib: float = 11.0,
) -> dict[int, str]:
    """
    Build a mildly asymmetric max_memory map for dual-GPU auto-sharded loading.

    Empirically on 2x32GB cards:
    - pure `device_map="auto"` overfills GPU0 during the first compressed-tensors
      decompression pass;
    - overly aggressive asymmetry pushes too much decompression pressure onto GPU1
      or causes disk offload.

    The target here is a middle ground around ~12GiB on GPU0 and ~22GiB on GPU1.
    This is a runtime-placement adapter only; it does not change HMO semantics.
    """
    gib = 1024 ** 3
    max_memory: dict[int, str] = {}
    for device_idx in range(torch.cuda.device_count()):
        total_bytes = torch.cuda.get_device_properties(device_idx).total_memory
        reserve = reserve_gib + (first_gpu_extra_reserve_gib if device_idx == 0 else 0.0)
        min_fraction = 0.35 if device_idx == 0 else 0.43
        usable_bytes = max(int(total_bytes - reserve * gib), int(total_bytes * min_fraction))
        usable_gib = max(1, int(usable_bytes // gib))
        max_memory[device_idx] = f"{usable_gib}GiB"
    return max_memory

# Model registry: name -> (subdir, dtype, load_kwargs)
MODEL_REGISTRY = {
    "qwen3.5-0.8b": {
        "path": "Qwen3.5-0.8B",
        "dtype": torch.bfloat16,
        "kwargs": {},
    },
    "qwen3.5-4b": {
        "path": "Qwen3.5-4B",
        "dtype": torch.bfloat16,
        "kwargs": {},
    },
    "qwen3.5-9b": {
        "path": "Qwen3.5-9B",
        "dtype": torch.bfloat16,
        "kwargs": {},
    },
    "qwen3.5-9b-gptq-int4": {
        "path": "Qwen3.5-9B-GPTQ-Int4",
        "dtype": None,
        "kwargs": {},
    },
    "qwen3.5-27b": {
        "path": "Qwen3.5-27B",
        "dtype": torch.bfloat16,
        "kwargs": {},
    },
    "qwen3.5-27b-gptq-int4": {
        "path": "Qwen3.5-27B-GPTQ-Int4",
        "dtype": None,
        "kwargs": {},
    },
    "kimi-linear-48b-gptq-int4": {
        "path": "Kimi-Linear-48B-A3B-Instruct-GPTQ-Int4",
        "dtype": None,
        "kwargs": {},
        "family": "kimi",
    },
}


def get_layer_types(config) -> list[str]:
    """Extract layer_types from model config (Qwen3.5 or Kimi-Linear)."""
    # Kimi-Linear: uses linear_attn_config with kda_layers / full_attn_layers
    if hasattr(config, "linear_attn_config") and config.linear_attn_config is not None:
        lac = config.linear_attn_config
        if isinstance(lac, dict):
            kda = set(lac.get("kda_layers", []))
            full = set(lac.get("full_attn_layers", []))
        else:
            kda = set(getattr(lac, "kda_layers", []))
            full = set(getattr(lac, "full_attn_layers", []))
        n_layers = config.num_hidden_layers
        types = []
        for i in range(n_layers):
            # Kimi config uses 1-based layer indices (is_kda_layer checks layer_idx+1)
            if (i + 1) in kda:
                types.append("linear_attention")
            elif (i + 1) in full:
                types.append("full_attention")
            else:
                types.append("full_attention")
        return types
    # Qwen3.5: uses layer_types directly
    text_cfg = getattr(config, "text_config", config)
    return list(text_cfg.layer_types)


def get_linear_attention_indices(config) -> list[int]:
    """Return indices of DeltaNet (linear_attention) layers."""
    return [i for i, t in enumerate(get_layer_types(config)) if t == "linear_attention"]


def get_full_attention_indices(config) -> list[int]:
    """Return indices of standard attention layers."""
    return [i for i, t in enumerate(get_layer_types(config)) if t == "full_attention"]


def load_model_and_tokenizer(
    model_name: str,
    device: str = "cuda",
    gpu_id: int = 1,
):
    """
    Load model and tokenizer from local SSD.

    Args:
        model_name: key in MODEL_REGISTRY (e.g. "qwen3.5-0.8b")
        device: "cuda" or "cpu"
        gpu_id: which GPU to use (default 1, since GPU 0 is occupied)
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    info = MODEL_REGISTRY[model_name]
    model_path = MODEL_ROOT / info["path"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Download it first.")

    logger.info(f"Loading {model_name} from {model_path}")

    # Fix gptqmodel triton patch before loading GPTQ models
    if info["dtype"] is None:
        _fix_gptqmodel_triton_patch()

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True
    )
    preload_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)

    load_kwargs = {
        "trust_remote_code": True,
        **info["kwargs"],
    }
    if info["dtype"] is not None:
        load_kwargs["dtype"] = info["dtype"]

    if device == "cuda":
        if info.get("family") == "kimi":
            load_kwargs["device_map"] = {"": f"cuda:{gpu_id}"}
            logger.info(
                f"{model_name}: using single-GPU Kimi placement on cuda:{gpu_id} "
                "(intended for the formal single-A100 E5 setup; no CPU/disk offload)"
            )
        elif info.get("multi_gpu"):
            load_kwargs["device_map"] = "auto"
            max_memory = _build_multi_gpu_max_memory()
            load_kwargs["max_memory"] = max_memory
            logger.info(
                f"{model_name}: using dual-GPU auto sharding with balanced max_memory={max_memory}"
            )
        else:
            load_kwargs["device_map"] = {"": f"cuda:{gpu_id}"}
    else:
        load_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    model.eval()

    config = model.config
    layer_types = get_layer_types(config)
    n_linear = sum(1 for t in layer_types if t == "linear_attention")
    n_full = sum(1 for t in layer_types if t == "full_attention")
    logger.info(f"Loaded {model_name}: {len(layer_types)} layers ({n_linear} DeltaNet + {n_full} Attention)")

    return model, tokenizer, config
