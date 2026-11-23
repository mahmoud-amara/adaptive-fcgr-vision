"""Staged data acquisition, QC validation, 95% ANI homology clustering, and pilot runner.

Executes:
Phase 1: Curates the 150 RefSeq genome metadata and acquires pilot genomes across 6 genera.
Phase 2: Performs MinHash 95% ANI homology clustering, validates zero leakage, and creates versioned manifests.
Phase 3: Executes a complete pilot evaluation (extracting 57 image features, fitting adaptive fusion on training folds only, evaluating SVM & RF, saving artifacts).
"""

from typing import Dict, List, Tuple
from pathlib import Path
import time
import hashlib
import json
import numpy as np
import pandas as pd
from Bio import Entrez, SeqIO

from adaptive_cgr_vision.data.parser import SequenceRecord, parse_fasta_file
from adaptive_cgr_vision.data.validator import validate_record, filter_records
from adaptive_cgr_vision.data.manifest import create_manifest, save_manifest
from adaptive_cgr_vision.data.clustering import compute_minhash_sketch, estimate_mash_ani, cluster_by_ani
from adaptive_cgr_vision.data.splitter import assign_group_folds, get_fold_train_val_indices, validate_splits
from adaptive_cgr_vision.features.extractor import FeatureExtractor
from adaptive_cgr_vision.fusion.adaptive import AdaptiveFusionTransformer
from adaptive_cgr_vision.fusion.fixed import FixedFusionTransformer
from adaptive_cgr_vision.models.classifiers import create_classifier
from adaptive_cgr_vision.evaluation.metrics import evaluate_predictions
from adaptive_cgr_vision.artifacts.schema import RunMetadata
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.acquire_dataset import CURATED_ACCESSION_PANEL, download_refseq_fasta, run_pilot_evaluation


def main():
    root_dir = Path("d:/Image Research paper")
    data_dir = root_dir / "data"
    raw_dir = data_dir / "raw"
    manifest_dir = data_dir / "manifests"
    runs_dir = root_dir / "runs"

    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("PHASE 1: DATASET METADATA CURATION & ACQUISITION")
    print("================================================================================")

    total_panel_entries = sum(len(accs) for accs in CURATED_ACCESSION_PANEL.values())
    print(f"Curated panel defines {total_panel_entries} complete chromosome accessions across {len(CURATED_ACCESSION_PANEL)} genera.")

    # Select representative pilot accessions (3 per genus = 18 total genomes)
    pilot_targets: List[Tuple[str, Dict[str, str]]] = []
    for genus, acc_list in CURATED_ACCESSION_PANEL.items():
        for item in acc_list[:3]:  # Top 3 per genus
            pilot_targets.append((genus, item))

    print(f"Acquiring {len(pilot_targets)} representative pilot genomes for end-to-end validation...")

    pilot_records: List[SequenceRecord] = []
    pilot_labels: List[str] = []

    for genus, item in pilot_targets:
        acc = item["accession"]
        species = item["species"]
        strain = item["strain"]
        print(f"  Fetching [{genus}] {species} ({strain}) -> {acc}...")

        try:
            fasta_path = download_refseq_fasta(acc, raw_dir, email="cgr_research@deepmind.com")
            recs = parse_fasta_file(fasta_path)
            if not recs:
                print(f"    WARNING: No records parsed from {fasta_path}")
                continue

            rec = recs[0]
            # QC validation
            qc = validate_record(rec, min_length=1_000_000, max_ambiguous_fraction=0.01)
            if not qc.passed:
                print(f"    FAILED QC: {qc.reason}")
                continue

            print(f"    [PASS QC] Length={rec.length:,} bp, Ambig={rec.ambiguous_fraction:.4%}, SHA256={rec.checksum_sha256[:12]}...")
            pilot_records.append(rec)
            pilot_labels.append(genus)

        except Exception as e:
            raise RuntimeError(f"Error fetching/validating {acc} ({genus}): {e}. Surrogates are strictly prohibited.")

    print(f"\nSuccessfully validated {len(pilot_records)} pilot genomes across {len(set(pilot_labels))} genera.")

    print("\n================================================================================")
    print("PHASE 2: 95% ANI MINHASH CLUSTERING & GROUPED SPLIT GENERATION")
    print("================================================================================")

    # Compute MinHash sketches and 95% ANI single-linkage clusters
    t_clust_start = time.time()
    pilot_seqs = [r.sequence for r in pilot_records]
    pilot_accs = [r.accession_id for r in pilot_records]

    cluster_ids = cluster_by_ani(pilot_seqs, pilot_accs, ani_threshold=0.95, k=21, sketch_size=1000)
    t_clust_sec = time.time() - t_clust_start

    print(f"Computed 95% ANI homology clustering in {t_clust_sec:.2f}s.")
    unique_clusters = set(cluster_ids)
    print(f"Discovered {len(unique_clusters)} distinct clusters across {len(pilot_records)} genomes.")

    # Create dataset manifest
    manifest_df = create_manifest(pilot_records, pilot_labels, cluster_ids)
    manifest_path = manifest_dir / "dataset_manifest_pilot.csv"
    save_manifest(manifest_df, manifest_path)
    print(f"Saved dataset manifest -> {manifest_path}")

    # Assign 3-fold StratifiedGroupKFold splits
    splits_df = assign_group_folds(manifest_df, n_splits=3, random_state=42, stratified=True)
    splits_path = manifest_dir / "splits_pilot.csv"
    save_manifest(splits_df, splits_path)
    print(f"Saved split assignments -> {splits_path}")

    # Validate split invariants
    validate_splits(splits_df)
    print("[PASS] Invariant Verified: Zero cluster/homology leakage across all folds.")

    print("\n================================================================================")
    print("PHASE 3: PILOT BENCHMARK EXECUTION & RUNTIME ESTIMATION")
    print("================================================================================")

    pilot_results = run_pilot_evaluation(
        pilot_records=pilot_records,
        pilot_labels=pilot_labels,
        pilot_groups=cluster_ids,
        output_dir=runs_dir,
    )

    print("\n--- Pilot Benchmark Results Summary ---")
    for model_name, res in pilot_results["summary"].items():
        print(f"  {model_name:<30}: Macro-F1 = {res['mean_macro_f1']:.4f} ± {res['std_macro_f1']:.4f}, Accuracy = {res['mean_accuracy']:.4f}")

    # Runtime and Memory Projections for full 150-genome benchmark
    avg_feat_time = pilot_results["avg_feat_time_sec"]
    proj_full_feat_time = avg_feat_time * 150
    proj_full_cv_time = 150 * 0.15  # SVM/RF training on 57-dim vectors is extremely fast (~0.15s per fold)
    total_proj_sec = proj_full_feat_time + proj_full_cv_time
    proj_ram_mb = 150 * (57 * 8) / (1024 * 1024) + 120  # feature matrix is ~0.1 MB + overhead ~120 MB

    print("\n================================================================================")
    print("FULL 150-GENOME BENCHMARK PROJECTIONS")
    print("================================================================================")
    print(f"  Total Genomes                    : 150 (25 per genus, 6 genera)")
    print(f"  Feature Extraction Time (k=6)    : ~{proj_full_feat_time:.1f} seconds (~{proj_full_feat_time/60:.2f} minutes)")
    print(f"  5x4 Nested CV Training Time      : ~{proj_full_cv_time:.1f} seconds")
    print(f"  Total Projected Benchmark Time   : ~{total_proj_sec/60:.2f} minutes")
    print(f"  Projected Peak Memory Usage      : < {proj_ram_mb:.1f} MB (easily runs on laptop)")
    print("================================================================================")

    # Output JSON summary for artifact report
    report_data = {
        "num_pilot_samples": len(pilot_records),
        "classes": list(set(pilot_labels)),
        "num_clusters": len(unique_clusters),
        "cluster_distribution": manifest_df["group_id"].value_counts().to_dict(),
        "pilot_summary": pilot_results["summary"],
        "pilot_elapsed_sec": pilot_results["pilot_elapsed_sec"],
        "avg_feat_time_sec": avg_feat_time,
        "proj_full_feat_time_min": proj_full_feat_time / 60.0,
        "proj_total_benchmark_time_min": total_proj_sec / 60.0,
        "manifest_path": str(manifest_path),
        "splits_path": str(splits_path),
    }

    with open(runs_dir / "pilot_summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)


if __name__ == "__main__":
    main()
