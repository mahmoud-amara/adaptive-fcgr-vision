"""End-to-end pipeline integration tests.

Verifies:
1. Sequence -> FCGR -> Features -> Fusion -> Classifier -> Evaluation -> Artifacts
2. Zero leakage across grouped folds
3. Baselines functionality (Raw FCGR, Smith-Waterman KNN)
4. Full run artifact serialization
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from adaptive_cgr_vision.data.parser import SequenceRecord
from adaptive_cgr_vision.data.validator import filter_records
from adaptive_cgr_vision.data.manifest import create_manifest, save_manifest, load_manifest
from adaptive_cgr_vision.data.splitter import assign_group_folds, get_fold_train_val_indices
from adaptive_cgr_vision.representations.fcgr import compute_fcgr
from adaptive_cgr_vision.features.extractor import FeatureExtractor
from adaptive_cgr_vision.fusion.fixed import FixedFusionTransformer
from adaptive_cgr_vision.fusion.adaptive import AdaptiveFusionTransformer
from adaptive_cgr_vision.models.classifiers import create_classifier
from adaptive_cgr_vision.models.baselines import RawFCGRBaseline, SmithWatermanKNNClassifier
from adaptive_cgr_vision.evaluation.metrics import evaluate_predictions
from adaptive_cgr_vision.evaluation.uncertainty import bootstrap_metric_ci, paired_difference_test
from adaptive_cgr_vision.artifacts.schema import RunMetadata
from adaptive_cgr_vision.artifacts.serialization import save_run_artifacts


@pytest.fixture
def synthetic_dataset():
    """Create a synthetic dataset of sequences across 3 classes and multiple groups."""
    rng = np.random.default_rng(123)
    bases = ["A", "C", "G", "T"]

    # Class 0: high GC rich
    # Class 1: high AT rich
    # Class 2: balanced
    class_probs = [
        [0.1, 0.4, 0.4, 0.1],  # GC rich
        [0.4, 0.1, 0.1, 0.4],  # AT rich
        [0.25, 0.25, 0.25, 0.25],  # balanced
    ]

    records = []
    labels = []
    groups = []

    for c_idx, probs in enumerate(class_probs):
        label_name = f"Species_{chr(65 + c_idx)}"
        for g_idx in range(4):  # 4 distinct strain groups per species
            grp_name = f"Group_{label_name}_{g_idx}"
            for rep in range(3):  # 3 sequences per group
                seq_len = rng.integers(1200, 1600)
                seq_arr = rng.choice(bases, size=seq_len, p=probs)
                # Introduce occasional Ns
                if rng.random() < 0.3:
                    seq_arr[10:13] = "N"
                seq_str = "".join(seq_arr)
                rec_id = f"ACC_{label_name}_{g_idx}_{rep}"

                rec = SequenceRecord(
                    accession_id=rec_id,
                    description=f"Synthetic sequence {rec_id}",
                    sequence=seq_str,
                    length=len(seq_str),
                    ambiguous_count=seq_str.count("N"),
                    ambiguous_fraction=seq_str.count("N") / len(seq_str),
                    checksum_sha256=f"hash_{rec_id}",
                )
                records.append(rec)
                labels.append(label_name)
                groups.append(grp_name)

    return records, labels, groups


class TestPipelineEndToEnd:
    def test_full_pipeline_execution(self, synthetic_dataset):
        records, labels, groups = synthetic_dataset

        # 1. Validation / QC
        passed_recs, qc_reports = filter_records(records, min_length=1000, max_ambiguous_fraction=0.05)
        assert len(passed_recs) == len(records)

        # 2. Manifest & Grouped Split
        manifest = create_manifest(passed_recs, labels, groups)
        manifest = assign_group_folds(manifest, n_splits=3, random_state=42, stratified=True)

        assert "fold" in manifest.columns
        assert set(manifest["fold"].unique()) == {0, 1, 2}

        # 3. Feature Extraction
        extractor = FeatureExtractor()
        feature_rows = []
        for rec in passed_recs:
            feats = extractor.extract_from_sequence(rec.sequence, k=4)
            feats["accession_id"] = rec.accession_id
            feature_rows.append(feats)

        feature_df = pd.DataFrame(feature_rows)
        feature_cols = [c for c in feature_df.columns if c != "accession_id"]
        X_all = feature_df[feature_cols].values
        y_all = manifest["label"].values

        # 4. Fold-by-fold Cross Validation with Adaptive Fusion
        train_idx, val_idx = get_fold_train_val_indices(manifest, fold_idx=0)

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        # Ensure no group overlap
        train_groups = set(manifest.iloc[train_idx]["group_id"])
        val_groups = set(manifest.iloc[val_idx]["group_id"])
        assert len(train_groups.intersection(val_groups)) == 0

        # Fit Adaptive Fusion strictly on training fold
        fusion = AdaptiveFusionTransformer(method="supervised_f_score")
        fusion.fit(X_train, y_train, feature_names=feature_cols)

        X_train_fused = fusion.transform(X_train)
        X_val_fused = fusion.transform(X_val)

        assert X_train_fused.shape == X_train.shape
        assert X_val_fused.shape == X_val.shape
        assert np.all(np.isfinite(X_train_fused))
        assert np.all(np.isfinite(X_val_fused))

        # Train Classifier (e.g. SVM RBF)
        clf = create_classifier("svm_rbf", random_state=42)
        clf.fit(X_train_fused, y_train)

        preds = clf.predict(X_val_fused)
        probs = clf.predict_proba(X_val_fused)

        # 5. Evaluate Metrics
        metrics = evaluate_predictions(
            y_true=y_val,
            y_pred=preds,
            y_prob=probs,
            classes=list(clf.classes_),
        )

        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "ece" in metrics
        assert "brier_score" in metrics

        # 6. Bootstrap CI & Statistical Testing
        point_est, ci_lo, ci_hi = bootstrap_metric_ci(y_val, preds, n_bootstraps=100)
        assert ci_lo <= point_est <= ci_hi

        # 7. Artifact Serialization
        with tempfile.TemporaryDirectory() as tmpdir:
            meta = RunMetadata(
                run_id="test_run_001",
                experiment_name="integration_test",
                model_name="svm_rbf",
                fusion_method="supervised_f_score",
            )
            pred_df = pd.DataFrame({
                "accession_id": manifest.iloc[val_idx]["accession_id"].values,
                "y_true": y_val,
                "y_pred": preds,
            })
            saved_path = save_run_artifacts(
                run_dir=tmpdir,
                metadata=meta,
                metrics=metrics,
                predictions_df=pred_df,
            )
            assert (Path(tmpdir) / "metadata.json").exists()
            assert (Path(tmpdir) / "metrics.json").exists()
            assert (Path(tmpdir) / "predictions.parquet").exists()


class TestBaselinesPipeline:
    def test_raw_fcgr_baseline(self, synthetic_dataset):
        records, labels, _ = synthetic_dataset
        seqs = [r.sequence for r in records[:20]]
        y = np.array(labels[:20])

        baseline = RawFCGRBaseline(k=4)
        baseline.fit(seqs, y)
        preds = baseline.predict(seqs)
        assert len(preds) == len(seqs)

    def test_smith_waterman_baseline(self, synthetic_dataset):
        records, labels, _ = synthetic_dataset
        # Short sub-sequences for quick test of SW KNN
        seqs = [r.sequence[:100] for r in records[:6]]
        y = np.array(labels[:6])

        sw_clf = SmithWatermanKNNClassifier()
        sw_clf.fit(seqs, y)
        preds = sw_clf.predict(seqs)
        assert len(preds) == len(seqs)
        assert set(preds).issubset(set(sw_clf.classes_))
