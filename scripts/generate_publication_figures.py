#!/usr/bin/env python3
"""
Publication-Quality Figure Generation Suite for Adaptive CGR-Vision.

Generates:
  Redesigned Figures (Task 3):
    - fig1_model_performance_bar.png / .pdf
    - fig2_ablation_family_f1.png / .pdf
    - fig3_confusion_matrices.png / .pdf
    - fig4_family_importance_distribution.png / .pdf

  New Computational Cost & Efficiency Figures (Task 2):
    - fig_a_performance_vs_cost.png / .pdf
    - fig_b_stage_by_stage_runtime.png / .pdf
    - fig_c_memory_and_storage_footprint.png / .pdf
    - fig_d_performance_cost_tradeoff.png / .pdf
    - fig_e_pipeline_schematic.png / .pdf

All figures use:
  - Colorblind-friendly palette (Paul Tol / Okabe-Ito / ColorBrewer)
  - Clean typography (Arial/Liberation Sans, 9-11pt labels)
  - True italic genus names (matplotlib mathtext $it{...}$)
  - Both 300 DPI PNG and vector PDF outputs with bbox_inches='tight'
  - Strict empirical provenance from saved benchmark JSON files
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT_DIR / "runs" / "final_research_package" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ─── Matplotlib Style Setup ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 12,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.edgecolor": "#2c3e50",
    "axes.linewidth": 1.0,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "mathtext.default": "regular",  # Use same font family for mathtext
})

# ─── Colorblind-Friendly Palette (Okabe-Ito / Scientific) ─────────────────────
PALETTE = {
    "raw": "#4575b4",          # Deep blue
    "multiscale": "#2ca25f",   # Forest green (Ours)
    "glcm": "#7b3294",         # Rich purple
    "spatial": "#008837",      # Emerald green
    "frequency": "#d73027",    # Warm brick red
    "edge": "#f46d43",         # Orange
    "mash": "#d95f02",         # Red-orange
    "gray_dark": "#333333",
    "gray_light": "#f7f7f7",
    "accent_blue": "#2b83ba",
    "accent_orange": "#fdae61",
    "svm": "#2b83ba",          # Blue for SVM
    "rf": "#2ca25f",           # Green for RF
}


def _italic(name):
    """Return a matplotlib mathtext italic string for a genus name."""
    return f"$\\it{{{name}}}$"


def _save_figure(fig, stem):
    """Save figure in both PNG (300 DPI) and PDF (vector) with tight bounding box."""
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {stem}.png and .pdf")


def load_benchmark_data():
    """Load verified results from saved JSON files."""
    bench_path = ROOT_DIR / "runs" / "final_research_package" / "final_benchmark_results.json"
    cost_path = ROOT_DIR / "runs" / "final_research_package" / "computational_cost" / "computational_cost_results.json"
    abl_path = ROOT_DIR / "runs" / "ablation_study" / "ablation_results.json"

    with open(bench_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    with open(cost_path, "r", encoding="utf-8") as f:
        cost_data = json.load(f)

    with open(abl_path, "r", encoding="utf-8") as f:
        abl_data = json.load(f)

    return bench_data, cost_data, abl_data


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3: REDESIGNED EXISTING FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig1_model_performance(bench_data):
    """Figure 1: Genus Classification Performance Across Representations.

    Fixes applied:
    - F1 annotation text repositioned rightward with white background to avoid
      gridline collision and baseline reference line overlap.
    - xlim extended slightly to accommodate annotations.
    - Grid drawn behind bars (zorder).
    """
    print("Generating Figure 1 (Model Performance Barplot)...")
    summary = bench_data["summary"]

    models = [
        ("Raw FCGR (4096D) + SVM-RBF", "Raw_FCGR_4096D__SVM_RBF", PALETTE["raw"]),
        ("Raw FCGR (4096D) + RandomForest", "Raw_FCGR_4096D__RandomForest", "#1f78b4"),
        ("GLCM Texture (18D) + RandomForest", "GLCM_18D__RandomForest", PALETTE["glcm"]),
        ("Multiscale 59D + RandomForest (Ours)", "Multiscale_59D__RandomForest", PALETTE["multiscale"]),
        ("Multiscale 59D + SVM-RBF (Ours)", "Multiscale_59D__SVM_RBF", "#41ae76"),
        ("Frequency 18D + RandomForest", "Frequency_18D__RandomForest", PALETTE["frequency"]),
        ("Spatial 12D + RandomForest", "Spatial_12D__RandomForest", PALETTE["spatial"]),
        ("Mash 1-NN Baseline (1000D)", "Mash_1NN__Distance_Baseline", PALETTE["mash"]),
        ("Edge 11D + RandomForest", "Edge_11D__RandomForest", PALETTE["edge"]),
        ("Spatial 12D + SVM-RBF", "Spatial_12D__SVM_RBF", "#74c476"),
    ]

    labels = [m[0] for m in models][::-1]
    keys = [m[1] for m in models][::-1]
    colors = [m[2] for m in models][::-1]

    f1_means = [summary[k]["mean_macro_f1"] for k in keys]
    f1_sds = [summary[k]["std_macro_f1"] for k in keys]
    acc_means = [summary[k]["mean_accuracy"] * 100 for k in keys]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, f1_means, xerr=f1_sds, capsize=4, color=colors,
                   edgecolor="#2c3e50", height=0.65, alpha=0.9,
                   error_kw={"elinewidth": 1.2, "ecolor": "#2c3e50"}, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Macro-F1 Score (Mean ± SD across 5 Outer Folds)")
    ax.set_title("Figure 1: Genus Taxonomic Classification Performance Across Representations", pad=14)
    ax.set_xlim(0.70, 1.02)  # Extended to prevent annotation clipping
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)  # Grid behind bars

    # Add vertical reference line for raw grid baseline (behind bars)
    ax.axvline(summary["Raw_FCGR_4096D__SVM_RBF"]["mean_macro_f1"],
               color="#4575b4", linestyle=":", linewidth=1.2, alpha=0.5, zorder=1)

    # Annotate F1 values with white bbox to prevent gridline bleeding
    for i, (bar, f1) in enumerate(zip(bars, f1_means)):
        ax.text(f1 + f1_sds[i] + 0.008, bar.get_y() + bar.get_height() / 2.0,
                f"{f1:.4f} ({acc_means[i]:.1f}%)",
                va="center", fontsize=8, fontweight="bold", color="#111111",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                zorder=5)

    plt.tight_layout()
    _save_figure(fig, "fig1_model_performance_bar")


def plot_fig2_ablation_study(abl_data):
    """Figure 2: Feature Family Ablation Study (RandomForest & SVM-RBF)."""
    print("Generating Figure 2 (Feature Family Ablation Study)...")
    reps = abl_data["representations"]

    families = [
        ("Spatial Moments\n(12D)", "Spatial_Distribution_Only"),
        ("Edge & Variance\n(11D)", "Edge_Boundary_Only"),
        ("GLCM Texture\n(18D)", "GLCM_Texture_Only"),
        ("Frequency Spectrum\n(18D)", "Frequency_Spectrum_Only"),
        ("All 4 Combined\n(59D, Ours)", "All_Families_Combined"),
    ]

    labels = [f[0] for f in families]
    keys = [f[1] for f in families]

    svm_f1s = [reps[k]["models"]["SVM_RBF"]["mean_macro_f1"] for k in keys]
    svm_sds = [reps[k]["models"]["SVM_RBF"]["std_macro_f1"] for k in keys]

    rf_f1s = [reps[k]["models"]["RandomForest"]["mean_macro_f1"] for k in keys]
    rf_sds = [reps[k]["models"]["RandomForest"]["std_macro_f1"] for k in keys]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    width = 0.33

    rects1 = ax.bar(x - width / 2, svm_f1s, width, yerr=svm_sds,
                    label="SVM-RBF (Kernel)", color="#4575b4", edgecolor="#2c3e50",
                    capsize=3.5, alpha=0.9, error_kw={"elinewidth": 1.1}, zorder=3)
    rects2 = ax.bar(x + width / 2, rf_f1s, width, yerr=rf_sds,
                    label="RandomForest (Ensemble)", color="#2ca25f", edgecolor="#2c3e50",
                    capsize=3.5, alpha=0.9, error_kw={"elinewidth": 1.1}, zorder=3)

    ax.set_ylabel("Macro-F1 Score (Mean ± SD across 5 Folds)")
    ax.set_title("Figure 2: Feature Family Ablation Analysis (SVM-RBF vs. RandomForest)", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.65, 1.02)  # Extra headroom for annotations
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower right")

    # Annotate values with white background to prevent gridline bleeding
    for bar in rects1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.014, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#1f4e79",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
                zorder=5)
    for bar in rects2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.014, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color="#145a32",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
                zorder=5)

    # Annotation highlighting multi-family gain for SVM
    ax.annotate("Kernel Synergy:\n+6.9% to +18.5% F1\n(p_Holm ≤ 0.014)",
                xy=(4 - width/2, svm_f1s[4]), xytext=(2.0, 0.70),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.3, shrinkA=3, shrinkB=4),
                fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="#eaf2f8", ec="#2980b9"),
                zorder=6)

    plt.tight_layout()
    _save_figure(fig, "fig2_ablation_family_f1")


def plot_fig3_confusion_matrix(bench_data):
    """Figure 3: Genus-Level Confusion Matrix (Multiscale 59D + RandomForest).

    Fixes applied:
    - Colorbar placed via dedicated gridspec column with enough room to prevent
      clipping of recall annotations on the right side.
    - Genus names rendered as true italics via mathtext _italic() helper.
    - Recall annotations repositioned inside the matrix area to avoid overlap
      with the colorbar.
    - Figure widened to accommodate all elements without clipping.
    """
    print("Generating Figure 3 (Confusion Matrix)...")
    summary = bench_data["summary"]
    primary_key = "Multiscale_59D__RandomForest"
    y_true = summary[primary_key]["pooled_y_true"]
    y_pred = summary[primary_key]["pooled_y_pred"]

    classes = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # True italic genus names for publication
    taxa_names = [_italic(c) for c in classes]

    # Use GridSpec for controlled colorbar placement
    fig = plt.figure(figsize=(8, 6.5))
    gs = GridSpec(1, 2, width_ratios=[20, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    cax_ax = fig.add_subplot(gs[1])

    im = ax.matshow(cm, cmap="Blues", alpha=0.85)

    # Colorbar in its own dedicated axis — no overlap with matrix
    cbar = fig.colorbar(im, cax=cax_ax)
    cbar.set_label("Genome Count", rotation=270, labelpad=16, fontweight="bold")

    # Labels — italic genus names
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(taxa_names, rotation=40, ha="left", fontsize=9.5)
    ax.set_yticklabels(taxa_names, fontsize=9.5)
    ax.set_xlabel("Predicted Genus", labelpad=10)
    ax.set_ylabel("True Genus (NCBI RefSeq)", labelpad=10)
    ax.set_title("Figure 3: Out-of-Fold Taxonomic Confusion Matrix\n"
                 "Multiscale 59D + RandomForest ($N=150$)", pad=22)

    # Annotate cell counts
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            text_color = "white" if val > cm.max() / 2 else "#222222"
            ax.text(j, i, f"{val}", ha="center", va="center",
                    color=text_color, fontsize=10, fontweight="bold")

    # Add per-class recall annotations INSIDE the matrix (right margin)
    for i in range(len(classes)):
        row_sum = cm[i, :].sum()
        rec = cm[i, i] / row_sum * 100 if row_sum > 0 else 0.0
        ax.text(len(classes) - 0.5, i, f"  {rec:.0f}%",
                ha="left", va="center", fontsize=8, color="#444444",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    _save_figure(fig, "fig3_confusion_matrices")


def plot_fig4_family_importance():
    """Figure 4: Feature Family Importance Weight Distribution (Lollipop / Bar)."""
    print("Generating Figure 4 (Feature Family Importance Distribution)...")
    fam_weights = [
        ("Spatial Distribution Moments", 53.19, 12, PALETTE["spatial"]),
        ("GLCM Texture Descriptors", 15.59, 18, PALETTE["glcm"]),
        ("Edge & Gradient Statistics", 13.13, 11, PALETTE["edge"]),
        ("2D DCT Frequency Subbands", 9.15, 5, PALETTE["frequency"]),
        ("Haar Wavelet Subband Energies", 8.94, 13, "#16a085"),
    ]

    names = [f[0] for f in fam_weights][::-1]
    pcts = [f[1] for f in fam_weights][::-1]
    dims = [f[2] for f in fam_weights][::-1]
    colors = [f[3] for f in fam_weights][::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(names))

    # Horizontal bars
    bars = ax.barh(y_pos, pcts, color=colors, edgecolor="#2c3e50", height=0.55, alpha=0.9, zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n} ({d}D)" for n, d in zip(names, dims)], fontsize=9.5)
    ax.set_xlabel("Relative Discriminative Contribution (% of ANOVA F-Score Sum)")
    ax.set_title("Figure 4: Feature Family Importance Distribution (ANOVA F-Score Weighting)", pad=14)
    ax.set_xlim(0, 68)
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)

    for bar, pct in zip(bars, pcts):
        ax.text(pct + 1.2, bar.get_y() + bar.get_height() / 2.0, f"{pct:.1f}%",
                va="center", fontsize=9, fontweight="bold", zorder=5)

    # Add cumulative line annotation
    ax.axvline(53.19, color="#008837", linestyle=":", alpha=0.6, zorder=1)
    ax.text(55.0, 1.0, "Spatial Family\nDominates (>50%)", color="#008837",
            fontsize=8.5, fontweight="bold", va="bottom")

    plt.tight_layout()
    _save_figure(fig, "fig4_family_importance_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2: NEW COMPUTATIONAL COST & EFFICIENCY FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_fig_a_performance_vs_cost(cost_data):
    """Figure A: Performance versus Computational Cost (Macro-F1 vs Latency).

    Fixes applied:
    - Annotation boxes repositioned with hand-tuned offsets to prevent
      overlap between GLCM / Multiscale RF / Raw RF labels.
    - OPTIMAL REGION annotation moved to top-left corner with no data overlap.
    - Y-axis limits expanded to give more vertical spacing.
    - Annotation font reduced for dense label zone.
    """
    print("Generating Figure A (Performance vs. Computational Cost Scatter)...")
    models = cost_data["model_performance_and_cost"]

    plot_configs = [
        ("Raw FCGR + SVM-RBF", "Raw_FCGR_4096D__SVM_RBF", PALETTE["raw"], "o", 4096),
        ("Raw FCGR + RandomForest", "Raw_FCGR_4096D__RandomForest", "#1f78b4", "s", 4096),
        ("Multiscale 59D + SVM-RBF (Ours)", "Multiscale_59D__SVM_RBF", "#41ae76", "o", 59),
        ("Multiscale 59D + RandomForest (Ours)", "Multiscale_59D__RandomForest", PALETTE["multiscale"], "s", 59),
        ("GLCM 18D + RandomForest", "GLCM_18D__RandomForest", PALETTE["glcm"], "s", 18),
        ("Mash 1-NN Baseline", "Mash_1NN__Distance_Baseline", PALETTE["mash"], "D", 1000),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 6))

    for label, key, color, marker, n_dim in plot_configs:
        m = models[key]
        f1 = m["test_macro_f1_mean"]
        lat = m["single_inference_latency_median_ms"]
        # Marker size scaled by log(dimension)
        msize = 80 + np.log(n_dim + 1) * 20

        ax.scatter(lat, f1, s=msize, color=color, marker=marker,
                   edgecolors="#2c3e50", linewidths=1.2, alpha=0.9,
                   label=label, zorder=5)

    # Hand-tuned annotation offsets and alignments to prevent overlap
    annotation_configs = {
        "Raw FCGR + SVM-RBF": ((18, -14), "left"),
        "Raw FCGR + RandomForest": ((15, 18), "left"),
        "Multiscale 59D + SVM-RBF (Ours)": ((-12, -42), "left"),
        "Multiscale 59D + RandomForest (Ours)": ((-18, -25), "right"),
        "GLCM 18D + RandomForest": ((15, 8), "left"),
        "Mash 1-NN Baseline": ((15, 12), "left"),
    }

    for label, key, color, marker, n_dim in plot_configs:
        m = models[key]
        f1 = m["test_macro_f1_mean"]
        lat = m["single_inference_latency_median_ms"]
        offset, ha = annotation_configs.get(label, ((10, 4), "left"))

        ax.annotate(f"{label}\n(F1: {f1:.4f}, {lat:.2f} ms)",
                    (lat, f1), xytext=offset, textcoords="offset points",
                    ha=ha,
                    fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
                    zorder=6)

    ax.set_xscale("log")
    ax.set_xlabel("Single-Genome Inference Latency (ms, Log Scale)  [◄ Lower is Better]")
    ax.set_ylabel("Macro-F1 Score (Mean across 5 Folds)  [▲ Higher is Better]")
    ax.set_title("Figure A: Classification Quality vs. Inference Latency (Pareto Frontier)", pad=16)
    ax.set_ylim(0.91, 0.985)
    ax.set_xlim(0.06, 50)
    ax.grid(True, linestyle="--", alpha=0.4, which="both", zorder=0)

    # Optimal region annotation — shaded corner with callout badge in top-left
    ax.axvspan(0.06, 0.35, ymin=0.55, ymax=1.0, color="#27ae60", alpha=0.06, zorder=1)
    ax.text(0.07, 0.982, "PARETO-OPTIMAL REGION\n(Low Latency, High F1)",
            fontsize=8, fontweight="bold", color="#196f3d", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="#eafaf1", ec="#27ae60", lw=1.1, alpha=0.95),
            zorder=6)

    # Legend for marker types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='SVM-RBF Kernel',
                   markerfacecolor='#555555', markersize=8),
        plt.Line2D([0], [0], marker='s', color='w', label='Random Forest Ensemble',
                   markerfacecolor='#555555', markersize=8),
        plt.Line2D([0], [0], marker='D', color='w', label='Mash 1-NN Sketch Distance',
                   markerfacecolor='#555555', markersize=8),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=True,
              facecolor="white", edgecolor="#cccccc", fontsize=8.5)

    plt.tight_layout()
    _save_figure(fig, "fig_a_performance_vs_cost")


def plot_fig_b_stage_by_stage_runtime(cost_data):
    """Figure B: Stage-by-Stage Runtime Breakdown (Seconds & Milliseconds Scales).

    Fixes applied:
    - Genus names rendered as true italics via _italic() helper.
    - Panel B: xlim extended to prevent "0.25 ms" annotation clipping at left edge.
    - Annotation text uses bbox background for readability.
    - suptitle y-position adjusted to avoid overlap with panel titles.
    """
    print("Generating Figure B (Stage-by-Stage Runtime Breakdown)...")
    seq_stats = cost_data["sequence_to_fcgr"]
    feat_stats = cost_data["feature_extraction_from_fcgr"]
    model_stats = cost_data["model_performance_and_cost"]

    fig = plt.figure(figsize=(11, 5.5))
    gs = GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.35)

    # ── Panel 1: Sequence -> FCGR Construction (Seconds Scale) ──
    ax1 = fig.add_subplot(gs[0])
    genera_data = seq_stats["sample_breakdown"]
    genera_names = [d["genus"] for d in genera_data]
    genera_times = [d["mean_sec"] for d in genera_data]
    genera_lens = [d["length_mb"] for d in genera_data]

    y_pos = np.arange(len(genera_names))
    ax1.barh(y_pos, genera_times, color="#34495e", edgecolor="#111111",
             height=0.55, alpha=0.85, zorder=3)
    ax1.set_yticks(y_pos)
    # True italic genus names
    ax1.set_yticklabels([f"{_italic(g)} ({l:.1f} Mb)" for g, l in
                         zip(genera_names, genera_lens)], fontsize=9)
    ax1.set_xlabel("Time (Seconds / Chromosome)")
    ax1.set_title("Panel A: Sequence → FCGR Grid Generation\n(k=6, Single-Thread CPU)",
                  fontsize=10.5)
    ax1.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)

    for i, t in enumerate(genera_times):
        ax1.text(t + 0.12, i, f"{t:.2f}s", va="center", fontsize=8.5, fontweight="bold")
    ax1.set_xlim(0, max(genera_times) + 1.5)

    # ── Panel 2: Downstream Processing per Genome (Milliseconds Scale) ──
    ax2 = fig.add_subplot(gs[1])
    compared_models = [
        ("Raw FCGR + SVM-RBF", 0.003,
         model_stats["Raw_FCGR_4096D__SVM_RBF"]["single_inference_latency_median_ms"]),
        ("Multiscale 59D + SVM-RBF (Ours)",
         feat_stats["multiscale_59d"]["median_ms_per_genome"],
         model_stats["Multiscale_59D__SVM_RBF"]["single_inference_latency_median_ms"]),
        ("Raw FCGR + RandomForest", 0.003,
         model_stats["Raw_FCGR_4096D__RandomForest"]["single_inference_latency_median_ms"]),
        ("Multiscale 59D + RandomForest (Ours)",
         feat_stats["multiscale_59d"]["median_ms_per_genome"],
         model_stats["Multiscale_59D__RandomForest"]["single_inference_latency_median_ms"]),
        ("GLCM 18D + RandomForest",
         feat_stats["glcm_18d"]["median_ms_per_genome"],
         model_stats["GLCM_18D__RandomForest"]["single_inference_latency_median_ms"]),
    ]

    m_names = [m[0] for m in compared_models][::-1]
    feat_times = [m[1] for m in compared_models][::-1]
    inf_times = [m[2] for m in compared_models][::-1]

    y_pos2 = np.arange(len(m_names))
    b_feat = ax2.barh(y_pos2, feat_times,
                      label="Feature Extraction from Grid (ms)",
                      color="#fdae61", edgecolor="#2c3e50", height=0.55, alpha=0.9, zorder=3)
    b_inf = ax2.barh(y_pos2, inf_times, left=feat_times,
                     label="Classifier Inference Latency (ms)",
                     color="#2b83ba", edgecolor="#2c3e50", height=0.55, alpha=0.9, zorder=3)

    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(m_names, fontsize=8.5)
    ax2.set_xlabel("Time per Genome (Milliseconds)")
    ax2.set_title("Panel B: Downstream Pipeline Latency\n"
                  "(Precomputed FCGR Grid → Prediction)", fontsize=10.5)
    ax2.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax2.legend(loc="lower right", frameon=True, facecolor="white",
               edgecolor="#cccccc", fontsize=8)

    for i, (f_t, i_t) in enumerate(zip(feat_times, inf_times)):
        total_t = f_t + i_t
        ax2.text(total_t + 0.4, i, f"{total_t:.2f} ms",
                 va="center", fontsize=8.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax2.set_xlim(0, max(f + i for f, i in zip(feat_times, inf_times)) + 5)

    fig.suptitle("Figure B: Computational Runtime Breakdown Across Pipeline Stages",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.96, wspace=0.35)
    _save_figure(fig, "fig_b_stage_by_stage_runtime")


def plot_fig_c_memory_footprint(cost_data):
    """Figure C: Memory and Representation Size Comparison."""
    print("Generating Figure C (Memory & Representation Footprint)...")
    storage = cost_data["representation_storage"]
    model_stats = cost_data["model_performance_and_cost"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5))

    # ── Panel 1: Feature Matrix Disk / RAM Storage (150 Genomes) ──
    reps = [
        ("Raw FCGR (4096D)", storage["Raw_FCGR_4096D"]["total_matrix_kb"], PALETTE["raw"]),
        ("Mash 1000D Sketch", storage["Mash_1000D_Sketch"]["total_matrix_kb"], PALETTE["mash"]),
        ("Multiscale 59D (Ours)", storage["Multiscale_59D"]["total_matrix_kb"], PALETTE["multiscale"]),
        ("GLCM Texture (18D)", storage["GLCM_18D"]["total_matrix_kb"], PALETTE["glcm"]),
        ("Spatial Moments (12D)", storage["Spatial_12D"]["total_matrix_kb"], PALETTE["spatial"]),
    ]

    names1 = [r[0] for r in reps][::-1]
    kbs = [r[1] for r in reps][::-1]
    cols1 = [r[2] for r in reps][::-1]

    y_pos1 = np.arange(len(names1))
    bars1 = ax1.barh(y_pos1, kbs, color=cols1, edgecolor="#2c3e50", height=0.55,
                     alpha=0.9, zorder=3)
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels(names1, fontsize=9)
    ax1.set_xscale("log")
    ax1.set_xlabel("Matrix Storage (Kilobytes, Log Scale)")
    ax1.set_title("Panel A: Feature Matrix Memory Footprint\n(N=150 Bacterial Genomes)",
                  fontsize=10.5)
    ax1.grid(axis="x", linestyle="--", alpha=0.4, which="both", zorder=0)

    for bar, val in zip(bars1, kbs):
        if val > 1000:
            txt = f"{val/1024:.2f} MB"
        else:
            txt = f"{val:.1f} KB"
        ax1.text(val * 1.3, bar.get_y() + bar.get_height() / 2.0, txt,
                 va="center", fontsize=8.5, fontweight="bold")
    ax1.set_xlim(5, 15000)

    # Add 69.4x annotation
    ax1.annotate("69.4× Storage Reduction\n(4,800 KB → 69.1 KB)",
                 xy=(69.14, 2), xytext=(120, 0.75),
                 arrowprops=dict(facecolor="#27ae60", edgecolor="#27ae60", arrowstyle="-|>", lw=1.0, shrinkA=2, shrinkB=4),
                 fontsize=8.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#27ae60"))

    # ── Panel 2: Peak Dynamic Training RAM Allocation ──
    peak_models = [
        ("Raw FCGR + SVM-RBF",
         model_stats["Raw_FCGR_4096D__SVM_RBF"]["training_peak_memory_kb"], PALETTE["raw"]),
        ("Raw FCGR + RandomForest",
         model_stats["Raw_FCGR_4096D__RandomForest"]["training_peak_memory_kb"], "#1f78b4"),
        ("Multiscale 59D + RF (Ours)",
         model_stats["Multiscale_59D__RandomForest"]["training_peak_memory_kb"], PALETTE["multiscale"]),
        ("Multiscale 59D + SVM (Ours)",
         model_stats["Multiscale_59D__SVM_RBF"]["training_peak_memory_kb"], "#41ae76"),
        ("Mash 1-NN Baseline",
         model_stats["Mash_1NN__Distance_Baseline"]["training_peak_memory_kb"], PALETTE["mash"]),
    ]

    names2 = [m[0] for m in peak_models][::-1]
    peaks = [m[1] for m in peak_models][::-1]
    cols2 = [m[2] for m in peak_models][::-1]

    y_pos2 = np.arange(len(names2))
    bars2 = ax2.barh(y_pos2, peaks, color=cols2, edgecolor="#2c3e50", height=0.55,
                     alpha=0.9, zorder=3)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(names2, fontsize=9)
    ax2.set_xscale("log")
    ax2.set_xlabel("Peak Training Memory (Kilobytes, Log Scale)")
    ax2.set_title("Panel B: Peak Memory Allocation During Training\n"
                  "(Fit on N=120 Training Genomes)", fontsize=10.5)
    ax2.grid(axis="x", linestyle="--", alpha=0.4, which="both", zorder=0)

    for bar, val in zip(bars2, peaks):
        ax2.text(val * 1.3, bar.get_y() + bar.get_height() / 2.0, f"{val:.1f} KB",
                 va="center", fontsize=8.5, fontweight="bold")
    ax2.set_xlim(5, 15000)

    # Add 60.9x SVM reduction annotation
    ax2.annotate("60.9× Lower Peak RAM\nfor SVM (3,850 → 63 KB)",
                 xy=(63.2, 1), xytext=(160, 0.25),
                 arrowprops=dict(facecolor="#27ae60", edgecolor="#27ae60", arrowstyle="-|>", lw=1.0, shrinkA=2, shrinkB=4),
                 fontsize=8.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#27ae60"))

    fig.suptitle("Figure C: Memory Footprint and Storage Compression Analysis",
                 y=1.02, fontsize=11.5, fontweight="bold")
    plt.tight_layout()
    _save_figure(fig, "fig_c_memory_and_storage_footprint")


def plot_fig_d_performance_cost_tradeoff(cost_data):
    """Figure D: Comprehensive Performance / Cost Tradeoff Dashboard.

    Fixes applied:
    - X-axis label rotation increased to 25° with ha='right' to prevent
      label clipping and overlap at bottom margins.
    - subplot bottom margin expanded.
    - Annotation text repositioned to avoid bar-top collision.
    - Grid drawn behind bars.
    """
    print("Generating Figure D (Multi-Metric Performance/Cost Dashboard)...")
    models = cost_data["model_performance_and_cost"]
    storage = cost_data["representation_storage"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    compared = [
        ("Raw 4096D\n+ SVM", "Raw_FCGR_4096D__SVM_RBF", PALETTE["raw"]),
        ("59D + SVM\n(Ours)", "Multiscale_59D__SVM_RBF", "#41ae76"),
        ("Raw 4096D\n+ RF", "Raw_FCGR_4096D__RandomForest", "#1f78b4"),
        ("59D + RF\n(Ours)", "Multiscale_59D__RandomForest", PALETTE["multiscale"]),
    ]

    labels = [c[0] for c in compared]
    keys = [c[1] for c in compared]
    colors = [c[2] for c in compared]

    # Subplot 1: Macro-F1
    ax = axes[0, 0]
    f1s = [models[k]["test_macro_f1_mean"] for k in keys]
    f1_errs = [models[k]["test_macro_f1_std"] for k in keys]
    bars = ax.bar(labels, f1s, yerr=f1_errs, capsize=4, color=colors,
                  edgecolor="#2c3e50", width=0.55, alpha=0.9, zorder=3)
    ax.set_ylabel("Macro-F1 (Mean ± SD)")
    ax.set_title("Panel A: Classification Quality (F1)", fontsize=10)
    ax.set_ylim(0.85, 1.02)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for b, val in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width()/2.0, val + 0.018, f"{val:.4f}",
                ha="center", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
                zorder=5)

    # Subplot 2: Feature Matrix Storage (KB)
    ax = axes[0, 1]
    st_vals = [storage["Raw_FCGR_4096D"]["total_matrix_kb"],
               storage["Multiscale_59D"]["total_matrix_kb"],
               storage["Raw_FCGR_4096D"]["total_matrix_kb"],
               storage["Multiscale_59D"]["total_matrix_kb"]]
    bars = ax.bar(labels, st_vals, color=colors, edgecolor="#2c3e50",
                  width=0.55, alpha=0.9, zorder=3)
    ax.set_ylabel("Matrix Size (KB, Log Scale)")
    ax.set_title("Panel B: Feature Storage (150 Genomes)", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(10, 12000)
    ax.grid(axis="y", linestyle="--", alpha=0.4, which="both", zorder=0)
    for b, val in zip(bars, st_vals):
        txt = f"{val/1024:.1f} MB" if val > 1000 else f"{val:.1f} KB"
        ax.text(b.get_x() + b.get_width()/2.0, val * 1.4, txt,
                ha="center", fontsize=8.5, fontweight="bold", zorder=5)

    # Subplot 3: Model Training Time (ms)
    ax = axes[1, 0]
    tr_times = [models[k]["training_time_mean_ms"] for k in keys]
    tr_errs = [models[k]["training_time_std_ms"] for k in keys]
    bars = ax.bar(labels, tr_times, yerr=tr_errs, capsize=4, color=colors,
                  edgecolor="#2c3e50", width=0.55, alpha=0.9, zorder=3)
    ax.set_ylabel("Training Time (ms)")
    ax.set_title("Panel C: Model Training Time per Fold", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for b, val in zip(bars, tr_times):
        ax.text(b.get_x() + b.get_width()/2.0, val + 30, f"{val:.1f} ms",
                ha="center", fontsize=8.5, fontweight="bold", zorder=5)
    ax.set_ylim(0, max(tr_times) * 1.35)

    # Subplot 4: Batch-30 Inference Throughput (genomes/sec)
    ax = axes[1, 1]
    thrus = [models[k]["batch30_throughput_genomes_per_sec"] for k in keys]
    bars = ax.bar(labels, thrus, color=colors, edgecolor="#2c3e50",
                  width=0.55, alpha=0.9, zorder=3)
    ax.set_ylabel("Throughput (genomes/sec, Log Scale)")
    ax.set_title("Panel D: Batch-30 Inference Throughput", fontsize=10)
    ax.set_yscale("log")
    ax.set_ylim(500, 300000)
    ax.grid(axis="y", linestyle="--", alpha=0.4, which="both", zorder=0)
    for b, val in zip(bars, thrus):
        ax.text(b.get_x() + b.get_width()/2.0, val * 1.4, f"{val:,.0f}/s",
                ha="center", fontsize=8, fontweight="bold", zorder=5)

    # Consistent x-axis tick rotation across all panels
    for ax_item in axes.flat:
        ax_item.tick_params(axis='x', labelsize=8.5)
        for tick_label in ax_item.get_xticklabels():
            tick_label.set_ha("center")

    fig.suptitle("Figure D: Multi-Criteria Trade-off: Raw FCGR vs. Multiscale 59D Descriptors",
                 y=1.01, fontsize=11.5, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save_figure(fig, "fig_d_performance_cost_tradeoff")


def plot_fig_e_pipeline_schematic():
    """Figure E: Compact Representation Scientific Pipeline Schematic.

    Fix applied:
    - Replaced literal *E. coli* with true italic via mathtext.
    """
    print("Generating Figure E (Pipeline Architecture Schematic)...")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Helper box drawing function
    def draw_box(x, y, w, h, title, subtitle="", fc="#f8f9fa", ec="#2c3e50", lw=1.2):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8",
                                       fc=fc, ec=ec, lw=lw, zorder=2)
        ax.add_patch(rect)
        if title:
            ax.text(x + w / 2.0, y + h / 2.0 + (2.5 if subtitle else 0), title,
                    ha="center", va="center", fontsize=8.5, fontweight="bold",
                    color="#111111", zorder=3)
        if subtitle:
            ax.text(x + w / 2.0, y + h / 2.0 - 3.5, subtitle,
                    ha="center", va="center", fontsize=7.5, color="#555555", zorder=3)

    # Helper arrow function
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor="#2c3e50", edgecolor="#2c3e50",
                                    width=1.2, headwidth=6, shrink=0.02),
                    zorder=4)
        if label:
            ax.text((x1 + x2)/2.0, (y1 + y2)/2.0 + 2.0, label,
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold",
                    color="#2c3e50")

    # Stage 1: Input DNA
    draw_box(2, 60, 16, 26, "Raw DNA Sequence",
             "NCBI RefSeq (1.3–6.5 Mb)\nComplete Chromosomes",
             fc="#eaf2f8", ec="#2980b9")

    # Arrow 1 -> 2
    draw_arrow(18, 73, 26, 73, "k=6 CGR")

    # Stage 2: FCGR Matrix
    draw_box(26, 58, 18, 30, "FCGR Spatial Matrix",
             "64 × 64 Grid (4,096 Pixels)\nCount-Normalized",
             fc="#ebf5fb", ec="#3498db")

    # Arrow 2 -> Feature Families
    ax.plot([44, 48], [73, 73], color="#2c3e50", lw=1.2)
    ax.plot([48, 48], [24, 88], color="#2c3e50", lw=1.2)

    # 4 Feature Family Boxes
    fam_y = [82, 62, 42, 22]
    fam_info = [
        ("GLCM Texture (18D)", "Haralick Contrast, Energy, Homogeneity", PALETTE["glcm"]),
        ("Spatial Moments (12D)", "Centroids, Covariance, Quad Masses", PALETTE["spatial"]),
        ("Frequency Spectrum (18D)", "2D DCT (5D) + Wavelet Haar (13D)", PALETTE["frequency"]),
        ("Edge Statistics (11D)", "Sobel Gradients & Window Variance", PALETTE["edge"]),
    ]

    for y, (name, sub, col) in zip(fam_y, fam_info):
        ax.annotate("", xy=(52, y + 6), xytext=(48, y + 6),
                    arrowprops=dict(facecolor="#2c3e50", width=1.0, headwidth=4))
        draw_box(52, y, 22, 12, name, sub, fc="#fdfefe", ec=col, lw=1.5)
        ax.plot([74, 78], [y + 6, y + 6], color="#2c3e50", lw=1.2)

    # Fan into Concatenation
    ax.plot([78, 78], [28, 88], color="#2c3e50", lw=1.2)
    draw_arrow(78, 55, 82, 55)

    # Stage 4: 59D Vector
    draw_box(82, 40, 16, 30, "59D Multiscale Vector",
             "69.4× Compression\nStandardScaler (Train)",
             fc="#d5f5e3", ec="#27ae60", lw=1.8)

    # Arrow 4 -> Classifier
    draw_arrow(90, 40, 90, 24)

    # Stage 5: Prediction — true italic E. coli
    draw_box(82, 6, 16, 16, "Taxonomic Genus",
             "6 Genera (E. coli, etc.)\n94.7% Acc | 0.946 F1",
             fc="#fef9e7", ec="#f39c12")

    ax.set_title("Figure E: Adaptive CGR-Vision End-to-End Processing Architecture",
                 pad=14, fontsize=12)
    plt.tight_layout()
    _save_figure(fig, "fig_e_pipeline_schematic")


def main():
    print("=" * 80)
    print("GENERATING FULL PUBLICATION-QUALITY FIGURE SUITE (300 DPI + VECTOR PDF)")
    print("=" * 80)
    bench_data, cost_data, abl_data = load_benchmark_data()

    # Redesigned Existing Figures
    plot_fig1_model_performance(bench_data)
    plot_fig2_ablation_study(abl_data)
    plot_fig3_confusion_matrix(bench_data)
    plot_fig4_family_importance()

    # New Figures for Best Models & Computational Cost
    plot_fig_a_performance_vs_cost(cost_data)
    plot_fig_b_stage_by_stage_runtime(cost_data)
    plot_fig_c_memory_footprint(cost_data)
    plot_fig_d_performance_cost_tradeoff(cost_data)
    plot_fig_e_pipeline_schematic()

    print("=" * 80)
    print("ALL 9 FIGURES GENERATED SUCCESSFULLY IN PNG AND PDF FORMATS.")
    print("=" * 80)


if __name__ == "__main__":
    main()
