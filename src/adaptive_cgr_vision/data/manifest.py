"""Data manifest generation, serialization, and validation.

Generates reproducible manifest tables with checksums, metadata, and split assignments.
"""

from typing import List, Optional, Union
from pathlib import Path
import pandas as pd
from adaptive_cgr_vision.data.parser import SequenceRecord


def create_manifest(
    records: List[SequenceRecord],
    labels: List[str],
    groups: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create a standardized pandas DataFrame manifest for sequence dataset.

    Parameters
    ----------
    records : List[SequenceRecord]
        List of QC-passed sequence records.
    labels : List[str]
        Class label (e.g. taxon/species) for each record.
    groups : Optional[List[str]]
        Cluster/Group ID for each record (used for leakage prevention).
        If None, accession_id is used as group.

    Returns
    -------
    pd.DataFrame
        Standardized manifest table.
    """
    if len(records) != len(labels):
        raise ValueError(f"Mismatch: {len(records)} records vs {len(labels)} labels")

    if groups is None:
        groups = [r.accession_id for r in records]
    elif len(groups) != len(records):
        raise ValueError(f"Mismatch: {len(records)} records vs {len(groups)} groups")

    data = []
    for rec, label, grp in zip(records, labels, groups):
        data.append(
            {
                "accession_id": rec.accession_id,
                "label": str(label),
                "group_id": str(grp),
                "length": rec.length,
                "ambiguous_count": rec.ambiguous_count,
                "ambiguous_fraction": rec.ambiguous_fraction,
                "checksum_sha256": rec.checksum_sha256,
                "description": rec.description,
            }
        )

    df = pd.DataFrame(data)
    return df


def save_manifest(manifest: pd.DataFrame, file_path: Union[str, Path]) -> None:
    """Save manifest to CSV or Parquet format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        manifest.to_parquet(path, index=False)
    else:
        manifest.to_csv(path, index=False)


def load_manifest(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load manifest from CSV or Parquet format."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
