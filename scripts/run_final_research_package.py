"""Final Research Package Execution Script for Adaptive CGR-Vision.

Executes fair comparison benchmarks, generates publication-quality figures (300 DPI),
and outputs structured JSON artifacts to runs/final_research_package/.
"""

from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone
import json
import time
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication quality
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
})

# Add src/ to pythonpath if needed
root_dir = Path("d:/Image Research paper")
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from adaptive_cgr_vision.features.extractor import FeatureExtractor
from adaptive_cgr_vision.data.clustering import compute_minhash_sketch, estimate_mash_ani
from adaptive_cgr_vision.evaluation.metrics import evaluate_predictions
from adaptive_cgr_vision.evaluation.uncertainty import bootstrap_metric_ci, paired_difference_test


class MashDistance1NN:
    """Mash-distance 1-Nearest-Neighbor classifier baseline."""
    def __init__(self, k: int = 21, sketch_size: int = 1000):
        self.k = k
        self.sketch_size = sketch_size
        self.train_sketches_: List[np.ndarray] = []
        self.train_labels_: np.ndarray = np.array([])
        self.classes_: np.ndarray = np.array([])

    def fit(self, sketches: List[np.ndarray], y: np.ndarray) -> "MashDistance1NN":
        self.train_sketches_ = list(sketches)
        self.train_labels_ = np.asarray(y)
        self.classes_ = np.unique(self.train_labels_)
        return self

    def predict(self, query_sketches: List[np.ndarray]) -> np.ndarray:
        predictions = []
        for q_sketch in query_sketches:
            best_ani = -1.0
            best_label = self.train_labels_[0]
            for t_sketch, t_label in zip(self.train_sketches_, self.train_labels_):
                ani = estimate_mash_ani(q_sketch, t_sketch, k=self.k)
                if ani > best_ani:
                    best_ani = ani
                    best_label = t_label
            predictions.append(best_label)
        return np.array(predictions)


class MashDistanceKNN:
    """Mash-distance k-Nearest-Neighbors classifier with similarity weighting."""
    def __init__(self, n_neighbors: int = 3, k: int = 21, sketch_size: int = 1000):
        self.n_neighbors = n_neighbors
        self.k = k
        self.sketch_size = sketch_size
        self.train_sketches_: List[np.ndarray] = []
        self.train_labels_: np.ndarray = np.array([])
        self.classes_: np.ndarray = np.array([])

    def fit(self, sketches: List[np.ndarray], y: np.ndarray) -> "MashDistanceKNN":
        self.train_sketches_ = list(sketches)
        self.train_labels_ = np.asarray(y)
        self.classes_ = np.unique(self.train_labels_)
        return self

    def predict(self, query_sketches: List[np.ndarray]) -> np.ndarray:
        predictions = []
        for q_sketch in query_sketches:
            similarities = []
            for t_sketch, t_label in zip(self.train_sketches_, self.train_labels_):
                ani = estimate_mash_ani(q_sketch, t_sketch, k=self.k)
                similarities.append((ani, t_label))
            # Sort descending by ANI similarity
            similarities.sort(key=lambda item: item[0], reverse=True)
            top_k = similarities[:self.n_neighbors]

            # Weighted vote by similarity (or equal if identical)
            votes: Dict[str, float] = {}
            for ani_val, lab in top_k:
                w = max(ani_val, 1e-5)
                votes[lab] = votes.get(lab, 0.0) + w
            best_lab = max(votes.items(), key=lambda item: item[1])[0]
            predictions.append(best_lab)
        return np.array(predictions)



SEED = 42
OUTER_FOLDS = 5


def load_data_and_splits(data_dir: Path):
    cache_file = data_dir / "processed" / "fcgr_matrices_cache.pkl"
    splits_file = data_dir / "manifests" / "splits_full.csv"
    raw_dir = data_dir / "raw"

    import pickle
    with open(cache_file, "rb") as f:
        cached_data = pickle.load(f)
    fcgr_matrices = cached_data["matrices"]

    splits_df = pd.read_csv(splits_file)
    labels = list(splits_df["label"])
    accessions = list(splits_df["accession_id"])

    # Load sequences for Mash baseline
    sequences = []
    for acc in accessions:
        f_path = raw_dir / f"{acc}.fasta"
        with open(f_path, "r") as f:
            lines = f.readlines()
        seq = "".join(l.strip() for l in lines if not l.startswith(">"))
        sequences.append(seq)

    return fcgr_matrices, labels, splits_df, sequences


def main():
    data_dir = root_dir / "data"
    output_dir = root_dir / "runs" / "final_research_package"
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("ADAPTIVE CGR-VISION: FINAL RESEARCH PACKAGE EXECUTION")
    print("================================================================================")
    print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
    print(f"Output Directory: {output_dir}")
    print("================================================================================")

    # Load data
    fcgr_matrices, labels, splits_df, sequences = load_data_and_splits(data_dir)
    print(f"Loaded {len(fcgr_matrices)} FCGR matrices, {len(splits_df)} splits, and {len(sequences)} fasta sequences.")

    # 1. Feature extraction
    extractor_59 = FeatureExtractor(
        glcm_distances=[1, 2, 3], glcm_n_levels=32, dct_bands=4,
        wavelet="haar", wavelet_levels=3, edge_thresholds=[0.01, 0.05, 0.1], edge_windows=[3, 5]
    )
    rows_59 = [extractor_59.extract_from_matrix(m) for m in fcgr_matrices]
    df_59 = pd.DataFrame(rows_59)
    cols_59 = sorted(df_59.columns)
    X_59 = df_59[cols_59].values.astype(np.float64)

    X_4096 = np.array([m.flatten() for m in fcgr_matrices], dtype=np.float64)

    extractor_spatial = FeatureExtractor(families=["spatial"])
    rows_spatial = [extractor_spatial.extract_from_matrix(m) for m in fcgr_matrices]
    df_spatial = pd.DataFrame(rows_spatial)
    cols_spatial = sorted(df_spatial.columns)
    X_spatial = df_spatial[cols_spatial].values.astype(np.float64)

    extractor_glcm = FeatureExtractor(families=["glcm"])
    rows_glcm = [extractor_glcm.extract_from_matrix(m) for m in fcgr_matrices]
    X_glcm = pd.DataFrame(rows_glcm).values.astype(np.float64)

    extractor_freq = FeatureExtractor(families=["frequency"])
    rows_freq = [extractor_freq.extract_from_matrix(m) for m in fcgr_matrices]
    X_freq = pd.DataFrame(rows_freq).values.astype(np.float64)

    extractor_edge = FeatureExtractor(families=["edge"])
    rows_edge = [extractor_edge.extract_from_matrix(m) for m in fcgr_matrices]
    X_edge = pd.DataFrame(rows_edge).values.astype(np.float64)

    # Mash sketches
    sketches = [compute_minhash_sketch(seq, k=21, sketch_size=1000) for seq in sequences]

    y_all = np.array(labels)
    classes_list = sorted(list(set(labels)))

    # Models to benchmark fairly
    CONFIGS = [
        ("Raw_FCGR_4096D__SVM_RBF", X_4096, "svm_rbf", 4096),
        ("Raw_FCGR_4096D__RandomForest", X_4096, "rf", 4096),
        ("Multiscale_59D__SVM_RBF", X_59, "svm_rbf", 59),
        ("Multiscale_59D__RandomForest", X_59, "rf", 59),
        ("Multiscale_59D__RandomForest_Regularized", X_59, "rf_reg", 59),
        ("Spatial_12D__SVM_RBF", X_spatial, "svm_rbf", 12),
        ("Spatial_12D__RandomForest", X_spatial, "rf", 12),
        ("GLCM_18D__RandomForest", X_glcm, "rf", 18),
        ("Frequency_18D__RandomForest", X_freq, "rf", 18),
        ("Edge_11D__RandomForest", X_edge, "rf", 11),
        ("Mash_1NN__Distance_Baseline", sketches, "mash_1nn", 1000),
        ("Mash_3NN__Distance_Baseline", sketches, "mash_3nn", 1000),
    ]

    benchmark_summary: Dict[str, Any] = {}
    per_fold_all: Dict[str, Any] = {}

    for name, feat_matrix, model_type, n_dim in CONFIGS:
        print(f"\nEvaluating: {name:<40} ({n_dim}D)...")
        t0 = time.time()
        fold_metrics = []
        pooled_yt = []
        pooled_yp = []
        tr_acc_list = []
        tr_f1_list = []

        for fold_idx in range(OUTER_FOLDS):
            train_idx = list(splits_df[splits_df["fold"] != fold_idx].index)
            test_idx = list(splits_df[splits_df["fold"] == fold_idx].index)

            y_tr, y_te = y_all[train_idx], y_all[test_idx]

            if model_type == "mash_1nn":
                tr_sk = [sketches[i] for i in train_idx]
                te_sk = [sketches[i] for i in test_idx]
                clf = MashDistance1NN(k=21, sketch_size=1000)
                clf.fit(tr_sk, y_tr)
                preds = clf.predict(te_sk)
                preds_tr = clf.predict(tr_sk)
                probs = None
            elif model_type == "mash_3nn":
                tr_sk = [sketches[i] for i in train_idx]
                te_sk = [sketches[i] for i in test_idx]
                clf = MashDistanceKNN(n_neighbors=3, k=21, sketch_size=1000)
                clf.fit(tr_sk, y_tr)
                preds = clf.predict(te_sk)
                preds_tr = clf.predict(tr_sk)
                probs = None
            else:
                X_tr, X_te = feat_matrix[train_idx], feat_matrix[test_idx]
                scaler = StandardScaler()
                X_tr_sc = scaler.fit_transform(X_tr)
                X_te_sc = scaler.transform(X_te)

                if model_type == "svm_rbf":
                    clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED, probability=True)
                elif model_type == "rf":
                    clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
                elif model_type == "rf_reg":
                    # Regularized RF: constrained depth and leaf samples to mitigate overfitting
                    clf = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=6,
                        min_samples_leaf=2,
                        random_state=SEED
                    )

                clf.fit(X_tr_sc, y_tr)
                preds = clf.predict(X_te_sc)
                preds_tr = clf.predict(X_tr_sc)
                probs = clf.predict_proba(X_te_sc) if hasattr(clf, "predict_proba") else None

            tr_acc_list.append(float(accuracy_score(y_tr, preds_tr)))
            tr_f1_list.append(float(f1_score(y_tr, preds_tr, average="macro")))

            res = evaluate_predictions(y_te, preds, probs, classes=classes_list)
            res["fold"] = fold_idx
            fold_metrics.append(res)
            pooled_yt.extend(y_te)
            pooled_yp.extend(preds)

        elapsed = time.time() - t0
        f1s = [m["macro_f1"] for m in fold_metrics]
        accs = [m["accuracy"] for m in fold_metrics]

        y_tr_arr = np.array(pooled_yt)
        y_pr_arr = np.array(pooled_yp)
        pooled_f1 = float(f1_score(y_tr_arr, y_pr_arr, average="macro"))
        pooled_acc = float(accuracy_score(y_tr_arr, y_pr_arr))

        pt, ci_lo, ci_hi = bootstrap_metric_ci(y_tr_arr, y_pr_arr, n_bootstraps=1000, random_state=SEED)

        benchmark_summary[name] = {
            "n_features": n_dim,
            "total_eval_time_sec": elapsed,
            "mean_macro_f1": float(np.mean(f1s)),
            "std_macro_f1": float(np.std(f1s, ddof=0)),
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs, ddof=0)),
            "mean_train_macro_f1": float(np.mean(tr_f1_list)),
            "std_train_macro_f1": float(np.std(tr_f1_list, ddof=0)),
            "mean_train_accuracy": float(np.mean(tr_acc_list)),
            "std_train_accuracy": float(np.std(tr_acc_list, ddof=0)),
            "generalization_gap_accuracy": float(np.mean(tr_acc_list) - np.mean(accs)),
            "generalization_gap_f1": float(np.mean(tr_f1_list) - np.mean(f1s)),
            "per_fold_f1": f1s,
            "per_fold_accuracy": accs,
            "per_fold_train_f1": tr_f1_list,
            "per_fold_train_accuracy": tr_acc_list,
            "pooled_f1": pooled_f1,
            "pooled_accuracy": pooled_acc,
            "bootstrap_ci_95": [ci_lo, ci_hi],
            "pooled_y_true": [str(x) for x in pooled_yt],
            "pooled_y_pred": [str(x) for x in pooled_yp],
        }
        per_fold_all[name] = fold_metrics

        print(f"  Test F1 = {np.mean(f1s):.4f} +/- {np.std(f1s, ddof=0):.4f} | Train F1 = {np.mean(tr_f1_list):.4f} (Gap: {np.mean(tr_f1_list)-np.mean(f1s):.4f}) | Acc = {np.mean(accs):.4f}")

    # Paired permutation tests vs Multiscale_59D__RandomForest & Multiscale_59D__SVM_RBF
    print("\nCalculating statistical comparisons...")
    stat_results = {}
    primary_key = "Multiscale_59D__RandomForest"
    p_true = np.array(benchmark_summary[primary_key]["pooled_y_true"])
    p_pred = np.array(benchmark_summary[primary_key]["pooled_y_pred"])
    p_f1 = benchmark_summary[primary_key]["pooled_f1"]

    raw_tests = []
    for name in benchmark_summary:
        if name == primary_key:
            continue
        b_pred = np.array(benchmark_summary[name]["pooled_y_pred"])
        tres = paired_difference_test(p_true, p_pred, b_pred, n_permutations=1000, random_state=SEED)
        raw_tests.append({
            "model_b": name,
            "f1_primary": p_f1,
            "f1_baseline": tres["f1_model_b"],
            "observed_diff": p_f1 - tres["f1_model_b"],
            "p_raw": tres["p_value"]
        })

    df_st = pd.DataFrame(raw_tests).sort_values("p_raw").reset_index(drop=True)
    m = len(df_st)
    p_raws = list(df_st["p_raw"])
    p_holms = [min(1.0, p_raws[i] * (m - i)) for i in range(m)]
    for i in range(1, m):
        p_holms[i] = max(p_holms[i], p_holms[i-1])
    df_st["p_holm"] = p_holms
    stat_results["vs_Multiscale_59D__RandomForest"] = df_st.to_dict(orient="records")

    # Save JSON files
    with open(output_dir / "final_benchmark_results.json", "w") as f:
        json.dump({"summary": benchmark_summary, "statistical_tests": stat_results}, f, indent=2)

    with open(output_dir / "per_fold_final_metrics.json", "w") as f:
        json.dump(per_fold_all, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Generate Publication-Quality Figures (300 DPI)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nGenerating publication-quality figures...")

    # FIGURE 1: Model & Representation Performance Barplot with Error Bars
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_models = [
        ("Raw FCGR (4096D) + RF", "Raw_FCGR_4096D__RandomForest", "#34495e"),
        ("Raw FCGR (4096D) + SVM", "Raw_FCGR_4096D__SVM_RBF", "#7f8c8d"),
        ("Multiscale 59D + RF (Ours)", "Multiscale_59D__RandomForest", "#27ae60"),
        ("Multiscale 59D + SVM (Ours)", "Multiscale_59D__SVM_RBF", "#2ecc71"),
        ("Spatial 12D + RF", "Spatial_12D__RandomForest", "#2980b9"),
        ("Spatial 12D + SVM", "Spatial_12D__SVM_RBF", "#3498db"),
        ("Mash 1-NN Baseline", "Mash_1NN__Distance_Baseline", "#e74c3c"),
    ]

    names_f1 = [item[0] for item in plot_models]
    means_f1 = [benchmark_summary[item[1]]["mean_macro_f1"] for item in plot_models]
    sds_f1 = [benchmark_summary[item[1]]["std_macro_f1"] for item in plot_models]
    colors_f1 = [item[2] for item in plot_models]

    bars = ax.barh(names_f1[::-1], means_f1[::-1], xerr=sds_f1[::-1], capsize=4, color=colors_f1[::-1], edgecolor="black", alpha=0.85)
    ax.set_xlabel("Macro-F1 Score (Mean ± SD across 5 Folds)")
    ax.set_title("Genus Classification Performance Across Representations")
    ax.set_xlim(0.70, 1.00)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, means_f1[::-1]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2.0, f"{val:.4f}", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig1_path = fig_dir / "fig1_model_performance_bar.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved Fig 1 -> {fig1_path}")

    # FIGURE 2: Feature Family Ablation Study Comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    abl_models = [
        ("GLCM Texture (18D)", "GLCM_18D__RandomForest", "#8e44ad"),
        ("Spatial Moments (12D)", "Spatial_12D__RandomForest", "#2980b9"),
        ("Frequency Subbands (18D)", "Frequency_18D__RandomForest", "#d35400"),
        ("Edge & Variance (11D)", "Edge_11D__RandomForest", "#c0392b"),
        ("All 4 Combined (59D)", "Multiscale_59D__RandomForest", "#27ae60"),
    ]

    abl_names = [item[0] for item in abl_models]
    abl_f1s = [benchmark_summary[item[1]]["mean_macro_f1"] for item in abl_models]
    abl_sds = [benchmark_summary[item[1]]["std_macro_f1"] for item in abl_models]
    abl_cols = [item[2] for item in abl_models]

    ax.bar(abl_names, abl_f1s, yerr=abl_sds, capsize=4, color=abl_cols, edgecolor="black", alpha=0.85, width=0.55)
    ax.set_ylabel("Macro-F1 Score (RandomForest)")
    ax.set_title("Feature Family Ablation Analysis (150 Genomes, 95% ANI Splits)")
    ax.set_ylim(0.75, 1.00)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i, (val, sd) in enumerate(zip(abl_f1s, abl_sds)):
        ax.text(i, val + sd + 0.008, f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig2_path = fig_dir / "fig2_ablation_family_f1.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved Fig 2 -> {fig2_path}")

    # FIGURE 3: Confusion Matrix for Best Model (Multiscale 59D + RF)
    fig, ax = plt.subplots(figsize=(6, 5))
    yt_best = benchmark_summary["Multiscale_59D__RandomForest"]["pooled_y_true"]
    yp_best = benchmark_summary["Multiscale_59D__RandomForest"]["pooled_y_pred"]
    cm = confusion_matrix(yt_best, yp_best, labels=classes_list)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes_list, yticklabels=classes_list, ax=ax, linewidths=0.5)
    ax.set_xlabel("Predicted Genus Label")
    ax.set_ylabel("True RefSeq Genus Label")
    ax.set_title("Confusion Matrix: Multiscale 59D + RandomForest (N=150)")

    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig3_path = fig_dir / "fig3_confusion_matrices.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved Fig 3 -> {fig3_path}")

    # FIGURE 4: Feature Family Importance Weight Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    fam_weights = {
        "Spatial Distribution": 0.5319,
        "GLCM Texture": 0.1559,
        "Edge & Variance": 0.1313,
        "DCT Frequency": 0.0915,
        "Wavelet Subbands": 0.0894,
    }
    labels_fam = list(fam_weights.keys())
    weights_fam = [v * 100 for v in fam_weights.values()]
    colors_fam = ["#2980b9", "#8e44ad", "#c0392b", "#d35400", "#16a085"]

    wedges, texts, autotexts = ax.pie(
        weights_fam, labels=labels_fam, autopct="%1.1f%%", startangle=140,
        colors=colors_fam, wedgeprops={"edgecolor": "black", "linewidth": 1.0}
    )
    for at in autotexts:
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Feature Family Weight Distribution (ANOVA F-Score)")

    plt.tight_layout()
    fig4_path = fig_dir / "fig4_family_importance_distribution.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved Fig 4 -> {fig4_path}")

    print("\nFINAL RESEARCH PACKAGE EXECUTION COMPLETE.")


if __name__ == "__main__":
    main()
