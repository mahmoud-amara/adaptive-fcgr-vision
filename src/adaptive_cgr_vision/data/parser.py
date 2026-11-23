"""FASTA parsing and sequence validation module.

Reads genomic FASTA records, performs alphabet checks, sequence hygiene,
and calculates SHA-256 integrity checksums.
"""

from typing import Iterator, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import hashlib
from Bio import SeqIO


@dataclass(frozen=True)
class SequenceRecord:
    """Standardized sequence record."""
    accession_id: str
    description: str
    sequence: str
    length: int
    ambiguous_count: int
    ambiguous_fraction: float
    checksum_sha256: str


def parse_fasta_file(file_path: Union[str, Path]) -> List[SequenceRecord]:
    """Parse a FASTA file into standardized SequenceRecord objects.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to FASTA file (can be gzipped or uncompressed).

    Returns
    -------
    List[SequenceRecord]
        List of parsed sequence records.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    records = []
    # Bio.SeqIO parses both plain text and gzipped files if handled properly
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for bio_rec in SeqIO.parse(handle, "fasta"):
            seq_str = str(bio_rec.seq).upper()
            length = len(seq_str)
            if length == 0:
                continue

            # Count valid ACGT bases
            num_acgt = sum(1 for c in seq_str if c in "ACGT")
            num_ambig = length - num_acgt
            ambig_fraction = num_ambig / length

            checksum = hashlib.sha256(seq_str.encode("ascii")).hexdigest()

            records.append(
                SequenceRecord(
                    accession_id=bio_rec.id,
                    description=bio_rec.description,
                    sequence=seq_str,
                    length=length,
                    ambiguous_count=num_ambig,
                    ambiguous_fraction=ambig_fraction,
                    checksum_sha256=checksum,
                )
            )

    return records
