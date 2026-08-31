"""
HMO Research — Canonical saturation detector aggregation.
Aggregates per-segment detector features from DeltaNetHookManager into
the frozen Phase-1 HMO saturation score.

Canonical detector:
- rho: write-to-retention ratio from exposed Qwen3.5 DeltaNet gates
- c: novelty collision on normalized write directions
- g_pressure: inverse retention pressure derived from the scalar decay gate
- warmup suppression and repeat-text filtering at the segment level
"""
import numpy as np
from .hooks import SegmentSignals


def compute_repeat_segment_mask(
    input_ids,
    segment_length: int,
    ngram_size: int = 4,
    repetition_threshold: float = 0.4,
) -> np.ndarray:
    """Detect degenerate repeated-text segments from token ids."""
    tokens = input_ids[0].detach().cpu().tolist()
    masks = []
    for start in range(0, len(tokens), segment_length):
        end = min(start + segment_length, len(tokens))
        segment = tokens[start:end]
        if len(segment) < max(ngram_size * 2, 8):
            masks.append(False)
            continue

        ngrams = [tuple(segment[i:i + ngram_size]) for i in range(len(segment) - ngram_size + 1)]
        if not ngrams:
            masks.append(False)
            continue

        repetition_ratio = 1.0 - (len(set(ngrams)) / max(len(ngrams), 1))
        masks.append(repetition_ratio >= repetition_threshold)
    return np.array(masks, dtype=bool)


def compute_segment_saturation(
    signals_per_layer: dict[int, SegmentSignals],
    alpha_rho: float = 0.4,
    alpha_c: float = 0.3,
    alpha_g: float = 0.3,
    segment_length: int = 512,
    warmup_tokens: int = 50,
    input_ids=None,
    repeat_ngram_size: int = 4,
    repeat_threshold: float = 0.4,
    repeat_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Aggregate per-layer per-segment signals into a single saturation score.
    Canonical segment aggregation:
    sigma_j = max_ell [alpha_rho * rho_bar_j^ell + alpha_c * c_j^ell + alpha_g * p_j^ell]

    where:
      - rho_bar is per-layer z-score-normalized rho passed through a sigmoid
      - c is per-layer min-max normalized novelty collision
      - p is per-layer min-max normalized inverse retention pressure

    Args:
        signals_per_layer: dict[layer_idx -> SegmentSignals]
        alpha_*: mixing weights
    Returns:
        sigma: [num_segments] saturation scores (higher = more saturated)
    """
    if not signals_per_layer:
        return np.array([])

    # Get number of segments from first layer
    first = next(iter(signals_per_layer.values()))
    n_segs = len(first.rho_max)

    if n_segs == 0:
        return np.array([])

    # Stack across layers: [n_layers, n_segs]
    rho_all = np.array([s.rho_max for s in signals_per_layer.values()])
    c_all = np.array([s.c_max for s in signals_per_layer.values()])
    g_all = np.array([s.g_mag_min for s in signals_per_layer.values()])

    # Normalize per layer so one layer's scale does not dominate the rest.
    def norm01_per_layer(x):
        xmin = x.min(axis=1, keepdims=True)
        xmax = x.max(axis=1, keepdims=True)
        denom = xmax - xmin
        denom[denom < 1e-8] = 1.0
        return (x - xmin) / denom

    def zscore_sigmoid_per_layer(x):
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std[std < 1e-8] = 1.0
        z = (x - mean) / std
        return 1.0 / (1.0 + np.exp(-z))

    rho_norm = zscore_sigmoid_per_layer(rho_all)
    c_norm = norm01_per_layer(c_all)
    # For g: lower retention = higher saturation pressure, so invert then normalize.
    g_pressure = norm01_per_layer(1.0 / (g_all + 1e-8))

    if repeat_mask is None and input_ids is not None:
        repeat_mask = compute_repeat_segment_mask(
            input_ids,
            segment_length=segment_length,
            ngram_size=repeat_ngram_size,
            repetition_threshold=repeat_threshold,
        )
    if repeat_mask is not None and len(repeat_mask) > 0:
        repeat_mask = repeat_mask[:n_segs]
        c_norm[:, repeat_mask] = 0.0

    # Weighted combination per layer per segment, then max across layers.
    combined = alpha_rho * rho_norm + alpha_c * c_norm + alpha_g * g_pressure

    if warmup_tokens > 0:
        warmup_scale = np.ones(n_segs, dtype=np.float32)
        for seg_idx in range(n_segs):
            start = seg_idx * segment_length
            end = start + segment_length
            if end <= warmup_tokens:
                warmup_scale[seg_idx] = 0.0
            elif start < warmup_tokens:
                warmup_scale[seg_idx] = float(end - warmup_tokens) / float(max(end - start, 1))
        combined = combined * warmup_scale[None, :]

    # Max across layers for each segment
    sigma = combined.max(axis=0)  # [n_segs]
    return sigma
