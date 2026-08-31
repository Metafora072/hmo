"""
E3 Analysis: Compute correlations and generate mechanism figures from E3 data.

Reads e3_mechanism.jsonl and produces:
1. phi vs sigma vs alpha correlation with refresh gain
2. Sub-signal ablation (rho-only, c-only, g-only)
3. Summary statistics

Usage:
    python experiments/phase2/e3_mechanism/analyze.py
"""
import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = (
    Path(os.environ.get("HMO_RESULTS_ROOT", str(PROJECT_ROOT / "experiments/results")))
    / "e3_mechanism"
)


def load_e3_data():
    path = RESULTS_DIR / "e3_mechanism.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def analyze():
    rows = load_e3_data()
    print(f"Loaded {len(rows)} samples")

    # Flatten: for each tested segment, collect (sigma, alpha, phi, rho, c, g, gain)
    all_sigma, all_alpha, all_phi = [], [], []
    all_rho, all_c, all_g = [], [], []
    all_gains = []

    for row in rows:
        sigma = np.array(row["sigma"])
        alpha = np.array(row["alpha"])
        phi = np.array(row["phi"])
        sigma_rho = np.array(row["sigma_rho"])
        sigma_c = np.array(row["sigma_c"])
        sigma_g = np.array(row["sigma_g"])
        gains = row["oracle_gains"]

        for seg_str, gain in gains.items():
            seg = int(seg_str)
            if seg < len(sigma):
                all_sigma.append(sigma[seg])
                all_alpha.append(alpha[seg])
                all_phi.append(phi[seg])
                all_rho.append(sigma_rho[seg] if seg < len(sigma_rho) else 0.0)
                all_c.append(sigma_c[seg] if seg < len(sigma_c) else 0.0)
                all_g.append(sigma_g[seg] if seg < len(sigma_g) else 0.0)
                all_gains.append(gain)

    all_sigma = np.array(all_sigma)
    all_alpha = np.array(all_alpha)
    all_phi = np.array(all_phi)
    all_rho = np.array(all_rho)
    all_c = np.array(all_c)
    all_g = np.array(all_g)
    all_gains = np.array(all_gains)

    print(f"\nTotal oracle-tested segments: {len(all_gains)}")
    print(f"Positive gain segments: {(all_gains > 0).sum()} ({(all_gains > 0).mean()*100:.1f}%)")

    # Correlations
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score

    binary_labels = (all_gains > 0).astype(int)
    has_both = len(np.unique(binary_labels)) >= 2

    signals = {
        "sigma": all_sigma,
        "alpha": all_alpha,
        "phi (sigma*alpha)": all_phi,
        "rho-only": all_rho,
        "c-only": all_c,
        "g-only": all_g,
    }

    print("\n" + "=" * 70)
    print(f"{'Signal':<20} {'Pearson':>8} {'Spearman':>9} {'AUC':>6}")
    print("=" * 70)

    results = {}
    for name, vals in signals.items():
        r_p, _ = pearsonr(vals, all_gains)
        r_s, _ = spearmanr(vals, all_gains)
        auc = roc_auc_score(binary_labels, vals) if has_both else 0.5
        results[name] = {"pearson": float(r_p), "spearman": float(r_s), "auc": float(auc)}
        print(f"{name:<20} {r_p:>8.3f} {r_s:>9.3f} {auc:>6.3f}")

    print("=" * 70)

    # Save analysis
    analysis = {
        "n_samples": len(rows),
        "n_tested_segments": len(all_gains),
        "n_positive_gain": int((all_gains > 0).sum()),
        "correlations": results,
    }
    out_path = RESULTS_DIR / "e3_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {out_path}")


if __name__ == "__main__":
    analyze()
