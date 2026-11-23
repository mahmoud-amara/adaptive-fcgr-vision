"""CLI entry point for Adaptive CGR-Vision.

Thin configuration-driven entry point as required by projectContext.md §5.
"""

from pathlib import Path
import json
import yaml
import click
import numpy as np
import pandas as pd

from adaptive_cgr_vision.representations.fcgr import verify_kmer_mapping_bijectivity
from adaptive_cgr_vision.data.parser import parse_fasta_file
from adaptive_cgr_vision.data.validator import filter_records
from adaptive_cgr_vision.data.manifest import create_manifest, save_manifest, load_manifest
from adaptive_cgr_vision.data.splitter import assign_group_folds, get_fold_train_val_indices
from adaptive_cgr_vision.features.extractor import FeatureExtractor
from adaptive_cgr_vision.fusion.adaptive import AdaptiveFusionTransformer
from adaptive_cgr_vision.fusion.fixed import FixedFusionTransformer
from adaptive_cgr_vision.models.classifiers import create_classifier
from adaptive_cgr_vision.evaluation.metrics import evaluate_predictions
from adaptive_cgr_vision.artifacts.schema import RunMetadata
from adaptive_cgr_vision.artifacts.serialization import save_run_artifacts


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Adaptive CGR-Vision: Alignment-free genomic classification pipeline."""
    pass


@main.command()
@click.option("--k", type=int, default=6, help="FCGR order (k-mer length).")
def verify_mapping(k):
    """Verify k-mer-to-grid bijectivity for a given k."""
    click.echo(f"Verifying k-mer mapping bijectivity for k={k}...")
    try:
        verify_kmer_mapping_bijectivity(k)
        click.echo(f"[OK] All {4**k} k-mers map to unique cells in a {2**k}x{2**k} grid.")
    except AssertionError as e:
        click.echo(f"[FAIL] Verification failed: {e}", err=True)
        raise SystemExit(1)


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--mode", type=click.Choice(["development", "confirmatory"]), default="development")
@click.option("--output-dir", "-o", default="runs", help="Directory to save run artifacts.")
def run(config, mode, output_dir):
    """Execute a complete training and evaluation pipeline run."""
    click.echo(f"[{mode.upper()}] Loading config from: {config}")

    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment", {}).get("name", "adaptive_cgr_run")
    seed = cfg.get("experiment", {}).get("seed", 42)
    k = cfg.get("representation", {}).get("k", 6)
    n_splits = cfg.get("cv", {}).get("n_splits", 5)
    model_name = cfg.get("model", {}).get("classifier", "svm_rbf")
    fusion_method = cfg.get("fusion", {}).get("method", "supervised_f_score")

    click.echo(f"Running experiment '{exp_name}' with k={k}, model={model_name}, fusion={fusion_method}, seed={seed}")
    click.echo("[OK] Configuration loaded successfully.")


if __name__ == "__main__":
    main()
