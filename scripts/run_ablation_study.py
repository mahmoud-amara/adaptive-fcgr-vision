"""Feature Family Ablation Study for Adaptive CGR-Vision.

Evaluates individual feature families vs combined multi-family representation:
1. GLCM Texture only (18 features)
2. Spatial Distribution only (12 features)
3. Frequency Spectrum only (18 features)
4. Edge & Boundary only (11 features)
5. All four families combined (59 features)

Reuses pre-computed FCGR matrices and existing outer-fold splits (splits_full.csv).
Outputs results to runs/ablation_study/ without overwriting previous benchmarks.
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

# Add src/ to pythonpath if executing as standalone
root_dir = Path("d:/Image Research paper")
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from adaptive_cgr_vision.features.extractor import FeatureExtractor
from adaptive_cgr_vision.evaluation.metrics import evaluate_predictions
from adaptive_cgr_vision.evaluation.uncertainty import bootstrap_metric_ci, paired_difference_test


SEED = 42
OUTER_FOLDS = 5


def load_data_and_splits(
    data_dir: Path,
) -> Tuple[List[np.ndarray], List[str], pd.DataFrame]:
    """Load cached FCGR matrices and splits manifest."""
    cache_file = data_dir / "processed" / "fcgr_matrices_cache.pkl"
    splits_file = data_dir / "manifests" / "splits_full.csv"

    if not cache_file.exists():
        raise FileNotFoundError(f"FCGR cache file not found at {cache_file}")
    if not splits_file.exists():
        raise FileNotFoundError(f"Splits file not found at {splits_file}")

    import pickle
    with open(cache_file, "rb") as f:
        cached_data = pickle.load(f)
    fcgr_matrices = cached_data["matrices"]

    splits_df = pd.read_csv(splits_file)
    labels = list(splits_df["label"])

    return fcgr_matrices, labels, splits_df


def run_family_ablation():
    data_dir = root_dir / "data"
    output_dir = root_dir / "runs" / "ablation_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("ADAPTIVE CGR-VISION: FEATURE FAMILY ABLATION STUDY")
    print("================================================================================")
    print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Scikit-Learn Version: {sklearn.__version__}")
    print(f"Seed: {SEED}")
    print(f"Output Directory: {output_dir}")
    print("================================================================================")

    # 1. Load data & splits
    fcgr_matrices, labels, splits_df = load_data_and_splits(data_dir)
    print(f"Loaded {len(fcgr_matrices)} FCGR matrices and {len(splits_df)} split assignments.")

    # Define representations to ablate
    REPRESENTATIONS = [
        ("GLCM_Texture_Only", ["glcm"]),
        ("Spatial_Distribution_Only", ["spatial"]),
        ("Frequency_Spectrum_Only", ["frequency"]),
        ("Edge_Boundary_Only", ["edge"]),
        ("All_Families_Combined", ["glcm", "spatial", "frequency", "edge"]),
    ]

    ablation_summary: Dict[str, Any] = {}
    per_fold_ablation: Dict[str, Any] = {}

    # We evaluate two standard classifiers per representation with fixed params:
    # 1. SVM-RBF (C=1.0, gamma='scale')
    # 2. RandomForest (n_estimators=100, random_state=42)
    CLASSIFIERS = [
        ("SVM_RBF", lambda: SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED)),
        ("RandomForest", lambda: RandomForestClassifier(n_estimators=100, random_state=SEED)),
    ]

    for rep_name, families in REPRESENTATIONS:
        print(f"\n" + "-" * 80)
        print(f"EVALUATING REPRESENTATION: {rep_name} (Families: {families})")
        print("-" * 80)

        # Extract features for this representation
        t_feat0 = time.time()
        extractor = FeatureExtractor(
            families=families,
            glcm_distances=[1, 2, 3],
            glcm_n_levels=32,
            dct_bands=4,
            wavelet="haar",
            wavelet_levels=3,
            edge_thresholds=[0.01, 0.05, 0.1],
            edge_windows=[3, 5],
        )

        feature_rows = [extractor.extract_from_matrix(m) for m in fcgr_matrices]
        df_feats = pd.DataFrame(feature_rows)
        feat_cols = sorted(df_feats.columns)
        X_all = df_feats[feat_cols].values.astype(np.float64)
        y_all = np.array(labels)
        elapsed_feat = time.time() - t_feat0

        n_dim = X_all.shape[1]
        print(f"Features Extracted: {n_dim} dimensions across {len(X_all)} samples ({elapsed_feat:.3f}s total)")

        ablation_summary[rep_name] = {
            "families": families,
            "n_features": n_dim,
            "feature_names": feat_cols,
            "extraction_time_sec": elapsed_feat,
            "models": {},
        }
        per_fold_ablation[rep_name] = {}

        for clf_label, clf_factory in CLASSIFIERS:
            fold_metrics_list = []
            pooled_y_true = []
            pooled_y_pred = []

            for fold_idx in range(OUTER_FOLDS):
                train_idx = list(splits_df[splits_df["fold"] != fold_idx].index)
                test_idx = list(splits_df[splits_df["fold"] == fold_idx].index)

                X_tr, y_tr = X_all[train_idx], y_all[train_idx]
                X_te, y_te = X_all[test_idx], y_all[test_idx]

                # Zero leakage scaler: fit ONLY on training fold
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_te_scaled = scaler.transform(X_te)

                # Fit classifier on training fold ONLY
                t_clf0 = time.time()
                clf = clf_factory()
                clf.fit(X_tr_scaled, y_tr)
                preds = clf.predict(X_te_scaled)
                probs = clf.predict_proba(X_te_scaled) if hasattr(clf, "predict_proba") else None
                elapsed_clf = time.time() - t_clf0

                eval_res = evaluate_predictions(y_te, preds, probs, classes=sorted(list(set(labels))))
                eval_res["fold"] = fold_idx
                eval_res["elapsed_sec"] = elapsed_clf
                fold_metrics_list.append(eval_res)

                pooled_y_true.extend(y_te)
                pooled_y_pred.extend(preds)

            # Aggregate per-fold metrics
            f1s = [m["macro_f1"] for m in fold_metrics_list]
            accs = [m["accuracy"] for m in fold_metrics_list]

            y_tr_arr = np.array(pooled_y_true)
            y_pr_arr = np.array(pooled_y_pred)
            pooled_f1 = float(f1_score(y_tr_arr, y_pr_arr, average="macro"))
            pooled_acc = float(accuracy_score(y_tr_arr, y_pr_arr))

            # Bootstrap CIs on pooled predictions (1000 resamples)
            pt, ci_lo, ci_hi = bootstrap_metric_ci(
                y_tr_arr, y_pr_arr, n_bootstraps=1000, confidence_level=0.95, random_state=SEED
            )

            ablation_summary[rep_name]["models"][clf_label] = {
                "mean_macro_f1": float(np.mean(f1s)),
                "std_macro_f1": float(np.std(f1s, ddof=0)),
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs, ddof=0)),
                "per_fold_f1": f1s,
                "per_fold_accuracy": accs,
                "pooled_f1": pooled_f1,
                "pooled_accuracy": pooled_acc,
                "bootstrap_ci_95": [ci_lo, ci_hi],
                "pooled_y_true": [str(x) for x in pooled_y_true],
                "pooled_y_pred": [str(x) for x in pooled_y_pred],
            }
            per_fold_ablation[rep_name][clf_label] = fold_metrics_list

            print(f"  [{clf_label:<12}] Mean Macro-F1 = {np.mean(f1s):.4f} +/- {np.std(f1s, ddof=0):.4f} | Pooled F1 = {pooled_f1:.4f} [95% CI: {ci_lo:.4f}, {ci_hi:.4f}] | Mean Acc = {np.mean(accs):.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Paired Statistical Significance Analysis vs All_Families_Combined
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PAIRED PERMUTATION TESTS (All_Families_Combined vs Single Families)")
    print("=" * 80)

    statistical_comparisons: Dict[str, Any] = {}

    for clf_label, _ in CLASSIFIERS:
        statistical_comparisons[clf_label] = []
        comb_true = np.array(ablation_summary["All_Families_Combined"]["models"][clf_label]["pooled_y_true"])
        comb_pred = np.array(ablation_summary["All_Families_Combined"]["models"][clf_label]["pooled_y_pred"])
        comb_f1 = ablation_summary["All_Families_Combined"]["models"][clf_label]["pooled_f1"]

        single_fam_tests = []
        for rep_name, _ in REPRESENTATIONS:
            if rep_name == "All_Families_Combined":
                continue
            fam_true = np.array(ablation_summary[rep_name]["models"][clf_label]["pooled_y_true"])
            fam_pred = np.array(ablation_summary[rep_name]["models"][clf_label]["pooled_y_pred"])

            test_res = paired_difference_test(comb_true, comb_pred, fam_pred, n_permutations=1000, random_state=SEED)
            diff = comb_f1 - test_res["f1_model_b"]
            single_fam_tests.append({
                "representation": rep_name,
                "f1_combined": comb_f1,
                "f1_single": test_res["f1_model_b"],
                "observed_diff": diff,
                "p_raw": test_res["p_value"],
            })

        # Apply Holm-Bonferroni correction (m=4 tests)
        df_comp = pd.DataFrame(single_fam_tests).sort_values("p_raw").reset_index(drop=True)
        m = len(df_comp)
        p_raw_vals = list(df_comp["p_raw"])
        p_holm_vals = [min(1.0, p_raw_vals[i] * (m - i)) for i in range(m)]
        for i in range(1, m):
            p_holm_vals[i] = max(p_holm_vals[i], p_holm_vals[i - 1])
        df_comp["p_holm"] = p_holm_vals

        statistical_comparisons[clf_label] = df_comp.to_dict(orient="records")

        print(f"\nClassifier: {clf_label}")
        for row in df_comp.to_dict(orient="records"):
            sig_raw = "YES" if row["p_raw"] < 0.05 else "NO"
            sig_holm = "YES" if row["p_holm"] < 0.05 else "NO"
            print(f"  All_Combined vs {row['representation']:<27}: Diff = {row['observed_diff']:+.4f} | p_raw = {row['p_raw']:.4f} ({sig_raw}) | p_holm = {row['p_holm']:.4f} ({sig_holm})")

    # ─────────────────────────────────────────────────────────────────────────
    # Save Artifacts
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SAVING ABLATION STUDY ARTIFACTS")
    print("=" * 80)

    results_path = output_dir / "ablation_results.json"
    clean_summary = {}
    for rep, data in ablation_summary.items():
        clean_summary[rep] = {
            "families": data["families"],
            "n_features": data["n_features"],
            "extraction_time_sec": data["extraction_time_sec"],
            "feature_names": data["feature_names"],
            "models": {},
        }
        for clf_name, mdata in data["models"].items():
            clean_summary[rep]["models"][clf_name] = {
                "mean_macro_f1": mdata["mean_macro_f1"],
                "std_macro_f1": mdata["std_macro_f1"],
                "mean_accuracy": mdata["mean_accuracy"],
                "std_accuracy": mdata["std_accuracy"],
                "per_fold_f1": mdata["per_fold_f1"],
                "per_fold_accuracy": mdata["per_fold_accuracy"],
                "pooled_f1": mdata["pooled_f1"],
                "pooled_accuracy": mdata["pooled_accuracy"],
                "bootstrap_ci_95": mdata["bootstrap_ci_95"],
            }

    export_json = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_genomes": len(fcgr_matrices),
        "outer_folds": OUTER_FOLDS,
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "command": "python -m scripts.run_ablation_study",
        "representations": clean_summary,
        "statistical_comparisons": statistical_comparisons,
    }

    with open(results_path, "w") as f:
        json.dump(export_json, f, indent=2)
    print(f"Saved summary -> {results_path}")

    per_fold_path = output_dir / "per_fold_ablation_metrics.json"
    with open(per_fold_path, "w") as f:
        json.dump(per_fold_ablation, f, indent=2)
    print(f"Saved per-fold metrics -> {per_fold_path}")

    manifest_path = output_dir / "ablation_manifest.json"
    manifest_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "outer_folds": OUTER_FOLDS,
        "dataset_manifest": str(data_dir / "manifests" / "dataset_manifest_full.csv"),
        "splits_file": str(data_dir / "manifests" / "splits_full.csv"),
        "fcgr_cache": str(data_dir / "processed" / "fcgr_matrices_cache.pkl"),
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "command_args": sys.argv,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Saved manifest -> {manifest_path}")

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_family_ablation()
