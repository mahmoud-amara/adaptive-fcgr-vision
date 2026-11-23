"""Sequence validation and quality control filter.

Implements QC rules:
- Minimum sequence length
- Maximum ambiguity fraction threshold
- Non-empty sequence
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from adaptive_cgr_vision.data.parser import SequenceRecord


@dataclass(frozen=True)
class QCResult:
    passed: bool
    reason: str
    record: SequenceRecord


def validate_record(
    record: SequenceRecord,
    min_length: int = 1000,
    max_ambiguous_fraction: float = 0.05,
) -> QCResult:
    """Validate a sequence record against quality control criteria.

    Parameters
    ----------
    record : SequenceRecord
        The parsed sequence record.
    min_length : int
        Minimum allowed sequence length in bp.
    max_ambiguous_fraction : float
        Maximum allowed fraction of ambiguous (non-ACGT) bases.

    Returns
    -------
    QCResult
        Validation outcome and reason.
    """
    if record.length < min_length:
        return QCResult(
            passed=False,
            reason=f"Length {record.length} bp is below minimum {min_length} bp",
            record=record,
        )

    if record.ambiguous_fraction > max_ambiguous_fraction:
        return QCResult(
            passed=False,
            reason=f"Ambiguity fraction {record.ambiguous_fraction:.4f} exceeds threshold {max_ambiguous_fraction:.4f}",
            record=record,
        )

    return QCResult(passed=True, reason="PASS", record=record)


def filter_records(
    records: List[SequenceRecord],
    min_length: int = 1000,
    max_ambiguous_fraction: float = 0.05,
) -> Tuple[List[SequenceRecord], List[QCResult]]:
    """Filter a list of sequence records according to QC criteria."""
    passed_records = []
    qc_reports = []

    for rec in records:
        res = validate_record(
            rec,
            min_length=min_length,
            max_ambiguous_fraction=max_ambiguous_fraction,
        )
        qc_reports.append(res)
        if res.passed:
            passed_records.append(rec)

    return passed_records, qc_reports
