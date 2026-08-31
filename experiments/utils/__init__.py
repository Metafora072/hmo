# HMO Research — Shared Utilities
from .model_loader import load_model_and_tokenizer, get_linear_attention_indices, get_full_attention_indices
from .hooks import DeltaNetHookManager, SegmentSignals
from .saturation import compute_segment_saturation
from .kv_ops import evict_kv_tokens, replace_with_skeleton, drop_segment, execute_refresh
from .hmo_controller import HMOController, HMOConfig, HMOResult
from .metrics import compute_exact_match, compute_f1, compute_correlation, LatencyTracker, get_peak_vram_mb
from .dataset_utils import EvalSample, make_needle_samples, load_longbench_subset
from .eval_harness import run_eval_suite, generate_and_evaluate
