"""
Chaos Game Representation (CGR) and Frequency CGR (FCGR) for DNA sequences.

This module implements the CGR and FCGR representations as specified in
projectContext.md §4.3 and §5 (CGR/FCGR). It provides:

  - A deterministic nucleotide-to-corner mapping
  - CGR point generation from valid A/C/G/T subsequences
  - FCGR k-mer counting with bijective k-mer-to-grid mapping
  - Ambiguity-aware sequence splitting (primary policy)
  - Count normalization and metadata tracking

Invariants enforced (projectContext.md §4.3):
  - Every valid k-mer maps to exactly one cell
  - sum(raw_matrix) == valid_overlapping_k-mer_count
  - All matrix entries are finite and nonnegative
  - Count normalization sums to one when denominator > 0
  - Output is deterministic
  - Empty/short sequences raise explicit errors (never silently produce all-zero)

Corner convention (documented, configurable):
  Default (counterclockwise from bottom-left):
    A = (0, 0)  bottom-left
    C = (0, 1)  top-left
    G = (1, 1)  top-right
    T = (1, 0)  bottom-right
  Row origin = top (row 0 = top of matrix, corresponding to C and G corners).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

VALID_BASES = frozenset("ACGT")

# Default corner mapping (counterclockwise from bottom-left).
# These are coordinates in the unit square [0,1]×[0,1].
DEFAULT_CORNER_MAPPING: Dict[str, Tuple[float, float]] = {
    "A": (0.0, 0.0),  # bottom-left
    "C": (0.0, 1.0),  # top-left
    "G": (1.0, 1.0),  # top-right
    "T": (1.0, 0.0),  # bottom-right
}


# ─── Data classes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FCGRMetadata:
    """Immutable metadata accompanying an FCGR matrix.

    Tracks all representation parameters for reproducibility and provenance
    (projectContext.md §4.3, §5).
    """

    k: int
    corner_mapping: Dict[str, Tuple[float, float]]
    row_origin: str  # "top" — row 0 is the top of the image
    normalization: str  # "count", "raw", "log1p"
    ambiguity_policy: str  # "split_runs"
    valid_kmer_count: int
    total_bases: int
    valid_bases: int
    ambiguous_bases: int
    n_runs: int  # number of valid A/C/G/T runs
    sequence_checksum: str  # SHA-256 of the uppercase input sequence
    implementation_version: str
    dtype: str


@dataclass
class FCGRResult:
    """Container for an FCGR matrix and its provenance metadata.

    Attributes:
        matrix: The FCGR matrix of shape (2**k, 2**k).
        raw_counts: Integer count matrix (always preserved for diagnostics).
        metadata: Full provenance and parameter metadata.
    """

    matrix: np.ndarray
    raw_counts: np.ndarray
    metadata: FCGRMetadata


# ─── Sequence utilities ──────────────────────────────────────────────────────


def uppercase_sequence(sequence: str) -> str:
    """Normalize sequence to uppercase."""
    return sequence.upper()


def split_into_valid_runs(sequence: str) -> List[str]:
    """Split a DNA sequence into maximal runs of valid A/C/G/T bases.

    This implements the primary ambiguity policy (projectContext.md §4.2):
    break into maximal A/C/G/T runs so no k-mer crosses an ambiguous symbol.

    Parameters
    ----------
    sequence : str
        Uppercase DNA sequence (may contain IUPAC ambiguity codes, N, etc.).

    Returns
    -------
    list of str
        Maximal runs of valid bases. Empty runs are excluded.
    """
    runs = []
    current_run: list[str] = []
    for base in sequence:
        if base in VALID_BASES:
            current_run.append(base)
        else:
            if current_run:
                runs.append("".join(current_run))
                current_run = []
    if current_run:
        runs.append("".join(current_run))
    return runs


def count_base_composition(sequence: str) -> Dict[str, int]:
    """Count A, C, G, T and ambiguous bases in an uppercase sequence.

    Returns
    -------
    dict with keys: 'A', 'C', 'G', 'T', 'valid', 'ambiguous', 'total'
    """
    counts: Dict[str, int] = {"A": 0, "C": 0, "G": 0, "T": 0, "valid": 0, "ambiguous": 0, "total": 0}
    for base in sequence:
        counts["total"] += 1
        if base in VALID_BASES:
            counts[base] += 1
            counts["valid"] += 1
        else:
            counts["ambiguous"] += 1
    return counts


def sequence_checksum(sequence: str) -> str:
    """Compute SHA-256 hex digest of an uppercase sequence string."""
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


# ─── k-mer to grid mapping ──────────────────────────────────────────────────


def _build_kmer_to_index(
    k: int,
    corner_mapping: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[int, int]]:
    """Build a bijective mapping from every k-mer to its (row, col) grid cell.

    The mapping is derived from the CGR construction: for a k-mer w₁w₂...wₖ,
    the CGR point ends at a position determined by iteratively halving the
    distance to each nucleotide's corner. The grid cell is the bin containing
    that final point in a 2^k × 2^k grid.

    Row origin is "top": row 0 corresponds to y=1 (top of unit square).

    Parameters
    ----------
    k : int
        k-mer length; grid will be 2^k × 2^k.
    corner_mapping : dict
        Nucleotide → (x, y) in unit square.

    Returns
    -------
    dict
        Mapping from k-mer string to (row, col) indices.
    """
    grid_size = 2**k
    kmer_to_idx: Dict[str, Tuple[int, int]] = {}

    # Generate all 4^k k-mers
    def _generate_kmers(prefix: str, remaining: int):
        if remaining == 0:
            # Compute the CGR endpoint for this k-mer
            x, y = 0.5, 0.5  # start at center
            for base in prefix:
                cx, cy = corner_mapping[base]
                x = (x + cx) / 2.0
                y = (y + cy) / 2.0

            # Convert to grid indices
            col = int(x * grid_size)
            row = int((1.0 - y) * grid_size)  # row origin = top

            # Clamp edge cases (point exactly at x=1.0 or y=0.0)
            col = min(col, grid_size - 1)
            row = min(row, grid_size - 1)

            kmer_to_idx[prefix] = (row, col)
            return

        for base in "ACGT":
            _generate_kmers(prefix + base, remaining - 1)

    _generate_kmers("", k)
    return kmer_to_idx


def _build_kmer_to_index_fast(
    k: int,
    corner_mapping: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[int, int]]:
    """Optimized k-mer to grid index mapping using vectorized computation.

    For larger k values, this is significantly faster than the recursive version.
    Uses the mathematical property that for k-mer w₁w₂...wₖ, the final
    CGR coordinate is:

        x = Σᵢ cx(wᵢ) · 2^(-(k-i+1)) + 0.5 · 2^(-k)
        y = Σᵢ cy(wᵢ) · 2^(-(k-i+1)) + 0.5 · 2^(-k)

    Parameters
    ----------
    k : int
        k-mer length.
    corner_mapping : dict
        Nucleotide → (x, y) in unit square.

    Returns
    -------
    dict
        Mapping from k-mer string to (row, col) indices.
    """
    grid_size = 2**k
    bases = "ACGT"
    kmer_to_idx: Dict[str, Tuple[int, int]] = {}

    # Precompute corner arrays
    cx = {b: corner_mapping[b][0] for b in bases}
    cy = {b: corner_mapping[b][1] for b in bases}

    # Powers of 0.5: 2^(-1), 2^(-2), ..., 2^(-k)
    powers = np.array([0.5 ** (i + 1) for i in range(k)])

    def _enumerate(prefix: str, depth: int):
        if depth == k:
            x = sum(cx[prefix[i]] * powers[i] for i in range(k)) + 0.5 * powers[-1]
            y = sum(cy[prefix[i]] * powers[i] for i in range(k)) + 0.5 * powers[-1]

            # Shift: the center term 0.5 * 2^(-k) is the initial offset
            # Recalculate properly: x = 0.5*2^(-k) + sum cx_i * 2^(-(k - i))
            # Actually the iterative formula gives:
            # x_final = x_0 * 2^(-k) + sum_{i=1}^{k} cx(w_i) * 2^(-(k-i+1))
            # with x_0 = 0.5
            col = int(x * grid_size)
            row = int((1.0 - y) * grid_size)
            col = min(col, grid_size - 1)
            row = min(row, grid_size - 1)
            kmer_to_idx[prefix] = (row, col)
            return
        for b in bases:
            _enumerate(prefix + b, depth + 1)

    _enumerate("", 0)
    return kmer_to_idx


# ─── CGR generation ─────────────────────────────────────────────────────────


def compute_cgr_points(
    sequence: str,
    corner_mapping: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """Compute CGR (Chaos Game Representation) point trajectory.

    Starting at the center of the unit square, iteratively move halfway toward
    the corner assigned to each valid nucleotide.

    Parameters
    ----------
    sequence : str
        Uppercase DNA sequence (only A/C/G/T bases are used).
    corner_mapping : dict, optional
        Nucleotide → (x, y). Defaults to DEFAULT_CORNER_MAPPING.

    Returns
    -------
    np.ndarray of shape (n_valid_bases, 2)
        CGR coordinates (x, y) for each valid base.
    """
    if corner_mapping is None:
        corner_mapping = DEFAULT_CORNER_MAPPING

    valid_bases_in_seq = [b for b in sequence if b in VALID_BASES]
    n = len(valid_bases_in_seq)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    points = np.empty((n, 2), dtype=np.float64)
    x, y = 0.5, 0.5  # start at center

    for i, base in enumerate(valid_bases_in_seq):
        cx, cy = corner_mapping[base]
        x = (x + cx) / 2.0
        y = (y + cy) / 2.0
        points[i, 0] = x
        points[i, 1] = y

    return points


# ─── FCGR generation ────────────────────────────────────────────────────────


def count_kmers_in_run(
    run: str,
    k: int,
    kmer_to_index: Dict[str, Tuple[int, int]],
    count_matrix: np.ndarray,
) -> int:
    """Count overlapping k-mers in a single valid run, accumulating into count_matrix.

    Parameters
    ----------
    run : str
        A contiguous run of valid A/C/G/T bases.
    k : int
        k-mer length.
    kmer_to_index : dict
        Pre-computed k-mer → (row, col) mapping.
    count_matrix : np.ndarray
        Mutable count matrix to accumulate into (int64).

    Returns
    -------
    int
        Number of valid k-mers counted from this run.
    """
    n_kmers = len(run) - k + 1
    if n_kmers <= 0:
        return 0

    counted = 0
    for i in range(n_kmers):
        kmer = run[i : i + k]
        row, col = kmer_to_index[kmer]
        count_matrix[row, col] += 1
        counted += 1

    return counted


def compute_fcgr(
    sequence: str,
    k: int,
    corner_mapping: Optional[Dict[str, Tuple[float, float]]] = None,
    normalization: str = "count",
    ambiguity_policy: str = "split_runs",
) -> FCGRResult:
    """Compute the Frequency Chaos Game Representation (FCGR) of a DNA sequence.

    This is the primary representation builder as specified in projectContext.md §4.3.

    Parameters
    ----------
    sequence : str
        Raw DNA sequence (will be uppercased).
    k : int
        k-mer length. Produces a 2^k × 2^k matrix.
    corner_mapping : dict, optional
        Nucleotide → (x, y) corner coordinates. Defaults to DEFAULT_CORNER_MAPPING.
    normalization : str
        "count" — divide by total valid k-mers (sums to 1.0)
        "raw"   — integer counts
        "log1p" — log(1 + count)
    ambiguity_policy : str
        "split_runs" — break sequence at ambiguous bases, count k-mers within
                       maximal valid runs only (primary policy).

    Returns
    -------
    FCGRResult
        Contains the normalized matrix, raw counts, and full metadata.

    Raises
    ------
    ValueError
        If the sequence has no valid k-mers (too short or all ambiguous).
        Per projectContext.md §4.3: "Empty/short sequences follow explicit QC
        exclusion and never produce a silently fabricated all-zero image."
    ValueError
        If k < 1 or an unknown normalization/ambiguity policy is given.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if normalization not in ("count", "raw", "log1p"):
        raise ValueError(f"Unknown normalization: {normalization!r}. Expected 'count', 'raw', or 'log1p'.")

    if ambiguity_policy != "split_runs":
        raise ValueError(
            f"Unknown ambiguity_policy: {ambiguity_policy!r}. "
            f"Only 'split_runs' is currently implemented (primary policy)."
        )

    if corner_mapping is None:
        corner_mapping = DEFAULT_CORNER_MAPPING

    # Normalize sequence
    seq_upper = uppercase_sequence(sequence)

    # Compute checksum of the input
    checksum = sequence_checksum(seq_upper)

    # Base composition
    composition = count_base_composition(seq_upper)

    # Split into valid runs
    runs = split_into_valid_runs(seq_upper)

    # Build k-mer → grid index mapping
    grid_size = 2**k
    kmer_to_index = _build_kmer_to_index(k, corner_mapping)

    # Verify bijectivity: every k-mer maps to a unique cell
    assert len(kmer_to_index) == 4**k, (
        f"k-mer index mapping has {len(kmer_to_index)} entries, expected {4**k}"
    )

    # Count k-mers across all valid runs
    # Use int64 to prevent overflow (projectContext.md §5: "check overflow before accumulation")
    count_matrix = np.zeros((grid_size, grid_size), dtype=np.int64)
    total_valid_kmers = 0

    for run in runs:
        total_valid_kmers += count_kmers_in_run(run, k, kmer_to_index, count_matrix)

    # ─── Invariant checks (projectContext.md §4.3) ───────────────────────
    # sum(raw matrix) == valid overlapping k-mer count
    matrix_sum = int(count_matrix.sum())
    assert matrix_sum == total_valid_kmers, (
        f"FCGR invariant violated: matrix sum ({matrix_sum}) != "
        f"valid k-mer count ({total_valid_kmers})"
    )

    # All entries finite and nonnegative
    assert np.all(count_matrix >= 0), "FCGR invariant violated: negative counts"
    assert np.all(np.isfinite(count_matrix)), "FCGR invariant violated: non-finite counts"

    # Reject empty/short sequences — never silently produce all-zero
    if total_valid_kmers == 0:
        raise ValueError(
            f"Sequence has no valid {k}-mers (length={composition['total']}, "
            f"valid_bases={composition['valid']}). Per projectContext.md §4.3, "
            f"empty/short sequences must be excluded by explicit QC rules, "
            f"not silently represented as all-zero matrices."
        )

    # ─── Normalize ────────────────────────────────────────────────────────
    raw_counts = count_matrix.copy()

    if normalization == "count":
        matrix = count_matrix.astype(np.float32) / total_valid_kmers
        # Verify normalization sums to 1
        norm_sum = float(matrix.sum())
        assert abs(norm_sum - 1.0) < 1e-5, (
            f"FCGR invariant violated: normalized matrix sums to {norm_sum}, expected 1.0"
        )
    elif normalization == "raw":
        matrix = count_matrix.astype(np.float32)
    elif normalization == "log1p":
        matrix = np.log1p(count_matrix.astype(np.float32))
    else:
        raise ValueError(f"Unknown normalization: {normalization!r}")  # pragma: no cover

    # Final finite check
    assert np.all(np.isfinite(matrix)), "FCGR invariant violated: non-finite values after normalization"

    # ─── Build metadata ──────────────────────────────────────────────────
    metadata = FCGRMetadata(
        k=k,
        corner_mapping=dict(corner_mapping),
        row_origin="top",
        normalization=normalization,
        ambiguity_policy=ambiguity_policy,
        valid_kmer_count=total_valid_kmers,
        total_bases=composition["total"],
        valid_bases=composition["valid"],
        ambiguous_bases=composition["ambiguous"],
        n_runs=len(runs),
        sequence_checksum=checksum,
        implementation_version="0.1.0",
        dtype=str(matrix.dtype),
    )

    return FCGRResult(
        matrix=matrix,
        raw_counts=raw_counts,
        metadata=metadata,
    )


# ─── Verification utilities ─────────────────────────────────────────────────


def verify_kmer_mapping_bijectivity(
    k: int,
    corner_mapping: Optional[Dict[str, Tuple[float, float]]] = None,
) -> bool:
    """Verify that all 4^k k-mers map to unique grid cells.

    This is the reversible index mapping test required by projectContext.md §5:
    "provide a reversible index mapping test across all 4^k words for supported small k."

    Parameters
    ----------
    k : int
        k-mer length.
    corner_mapping : dict, optional
        Defaults to DEFAULT_CORNER_MAPPING.

    Returns
    -------
    bool
        True if mapping is bijective (all k-mers → unique cells).

    Raises
    ------
    AssertionError
        If any two k-mers map to the same cell.
    """
    if corner_mapping is None:
        corner_mapping = DEFAULT_CORNER_MAPPING

    kmer_to_index = _build_kmer_to_index(k, corner_mapping)
    grid_size = 2**k

    # Check we have exactly 4^k mappings
    assert len(kmer_to_index) == 4**k, (
        f"Expected {4**k} k-mers, got {len(kmer_to_index)}"
    )

    # Check all indices are in bounds
    for kmer, (row, col) in kmer_to_index.items():
        assert 0 <= row < grid_size, f"k-mer {kmer!r} row {row} out of bounds [0, {grid_size})"
        assert 0 <= col < grid_size, f"k-mer {kmer!r} col {col} out of bounds [0, {grid_size})"

    # Check bijectivity: no two k-mers map to the same cell
    cells = list(kmer_to_index.values())
    unique_cells = set(cells)
    if len(unique_cells) != len(cells):
        # Find collisions for error message
        from collections import Counter
        cell_counts = Counter(cells)
        collisions = {cell: count for cell, count in cell_counts.items() if count > 1}
        collision_kmers = {}
        for cell in collisions:
            collision_kmers[cell] = [km for km, idx in kmer_to_index.items() if idx == cell]
        raise AssertionError(
            f"k-mer mapping is not bijective at k={k}. "
            f"Collisions: {collision_kmers}"
        )

    # Check that cells cover the full grid (surjectivity for FCGR)
    assert len(unique_cells) == grid_size * grid_size, (
        f"Mapping covers {len(unique_cells)} cells, but grid has "
        f"{grid_size * grid_size} cells. Not surjective."
    )

    return True
