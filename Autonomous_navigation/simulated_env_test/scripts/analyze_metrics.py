#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PRED_DIR = os.path.join(PKG_ROOT, "results", "predictions")
OUT_DIR = os.path.join(PKG_ROOT, "results")

def analyze_benchmarks():
    eval_files = sorted(glob.glob(os.path.join(PRED_DIR, "*_eval.npz")))
    records = []

    for f in eval_files:
        data = np.load(f)
        pred_depth = data["pred_depth"]
        gt_depth = data["gt_depth"]
        mask = data["mask"]
        
        # Image-wide valid depth mask (excluding sky/infinity)
        valid = (gt_depth > 0.1) & (gt_depth < 9.8) & (pred_depth > 0.1)
        
        # Image-wide metrics
        abs_rel = np.mean(np.abs(pred_depth[valid] - gt_depth[valid]) / gt_depth[valid])
        rmse = np.sqrt(np.mean((pred_depth[valid] - gt_depth[valid]) ** 2))
        ratio = np.maximum(pred_depth[valid] / gt_depth[valid], gt_depth[valid] / pred_depth[valid])
        delta1 = np.mean(ratio < 1.25) * 100.0

        records.append({
            "shape": str(data["shape"]),
            "nominal_dist": float(data["nominal_dist"]),
            "y_offset": float(data["y_offset"]),
            "gt_dist": float(data["gt_dist"]),
            "pred_dist": float(data["pred_dist"]),
            "image_abs_rel": float(abs_rel),
            "image_rmse": float(rmse),
            "image_delta1": float(delta1)
        })

    df = pd.DataFrame(records)

    # Group by experimental configuration
    grouped = df.groupby(["shape", "nominal_dist", "y_offset"]).agg(
        gt_dist_mean=("gt_dist", "mean"),
        pred_dist_mean=("pred_dist", "mean"),
        pred_dist_std=("pred_dist", "std"),
        abs_rel_mean=("image_abs_rel", "mean"),
        rmse_mean=("image_rmse", "mean"),
        delta1_mean=("image_delta1", "mean")
    ).reset_index()

    grouped["abs_error"] = np.abs(grouped["pred_dist_mean"] - grouped["gt_dist_mean"])
    grouped["rel_error_pct"] = (grouped["abs_error"] / grouped["gt_dist_mean"]) * 100.0
    grouped["flag_exceeds_10pct"] = grouped["rel_error_pct"] > 10.0

    csv_path = os.path.join(OUT_DIR, "benchmark_metrics.csv")
    grouped.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")

    # Scale fit analysis: y = alpha * x + beta
    x = grouped["gt_dist_mean"].values
    y = grouped["pred_dist_mean"].values
    poly = np.polyfit(x, y, 1)
    alpha, beta = poly[0], poly[1]

    print("\n============== SCALE & DRIFT REGRESSION ==============")
    print(f"Fit equation: Predicted = {alpha:.4f} * GroundTruth + {beta:.4f}")
    if abs(alpha - 1.0) < 0.05 and abs(beta) < 0.10:
        print("Verdict: Metrics reflect accurate 1:1 metric scale.")
    elif abs(beta) < 0.15:
        print(f"Verdict: Constant scale factor error detected (Multiplier: {alpha:.4f}).")
    else:
        print("Verdict: Distance-dependent non-linear/affine drift detected.")
    print("=======================================================\n")

    # Generate Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Predicted vs Ground Truth with y = x
    ax = axes[0]
    for shape in grouped["shape"].unique():
        sub = grouped[grouped["shape"] == shape]
        ax.errorbar(
            sub["gt_dist_mean"], sub["pred_dist_mean"],
            yerr=sub["pred_dist_std"], fmt='o', label=f"{shape}", capsize=4
        )
    max_d = max(x.max(), y.max()) + 1.0
    ax.plot([0, max_d], [0, max_d], 'k--', label="Ideal (y = x)")
    ax.plot(x, alpha * x + beta, 'r:', label=f"Fit: {alpha:.2f}x + {beta:.2f}")
    ax.set_title("UniDepth Metric Prediction vs. Ground Truth")
    ax.set_xlabel("Ground Truth Distance [m]")
    ax.set_ylabel("Predicted Distance [m]")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # Plot 2: Relative Error vs Ground Truth Distance
    ax2 = axes[1]
    for shape in grouped["shape"].unique():
        sub = grouped[grouped["shape"] == shape]
        ax2.plot(sub["gt_dist_mean"], sub["rel_error_pct"], marker='s', label=f"{shape}")
    ax2.axhline(10.0, color='red', linestyle='--', label="10% Error Threshold")
    ax2.set_title("Relative Depth Error vs. Distance")
    ax2.set_xlabel("Ground Truth Distance [m]")
    ax2.set_ylabel("Relative Error [%]")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plot_path = os.path.join(OUT_DIR, "unidepth_accuracy_curves.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"Validation plots saved to {plot_path}")

if __name__ == "__main__":
    analyze_benchmarks()
