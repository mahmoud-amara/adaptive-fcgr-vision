"""Data loading, QC validation, manifest management, split construction, and homology clustering."""

from adaptive_cgr_vision.data.parser import SequenceRecord, parse_fasta_file
from adaptive_cgr_vision.data.validator import QCResult, validate_record, filter_records
from adaptive_cgr_vision.data.manifest import create_manifest, save_manifest, load_manifest
from adaptive_cgr_vision.data.splitter import (
    assign_group_folds,
    validate_splits,
    get_fold_train_val_indices,
)
from adaptive_cgr_vision.data.clustering import (
    compute_minhash_sketch,
    estimate_mash_ani,
    cluster_by_ani,
)

__all__ = [
    "SequenceRecord",
    "parse_fasta_file",
    "QCResult",
    "validate_record",
    "filter_records",
    "create_manifest",
    "save_manifest",
    "load_manifest",
    "assign_group_folds",
    "validate_splits",
    "get_fold_train_val_indices",
    "compute_minhash_sketch",
    "estimate_mash_ani",
    "cluster_by_ani",
]
