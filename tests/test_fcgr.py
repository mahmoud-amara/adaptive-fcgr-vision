"""
Comprehensive tests for the CGR/FCGR representation module.

Tests cover:
  - Sequence utilities (uppercase, splitting, composition, checksum)
  - k-mer-to-grid bijectivity for k=1..5
  - CGR point generation
  - FCGR invariants (projectContext.md §4.3):
      * sum(raw) == valid k-mer count
      * all entries finite and nonnegative
      * normalized matrix sums to 1.0
      * deterministic output
      * empty/short sequences raise ValueError
  - Ambiguity handling (no cross-boundary k-mers)
  - Normalization modes
  - Edge cases
"""

import hashlib

import numpy as np
import pytest

from adaptive_cgr_vision.representations.fcgr import (
    DEFAULT_CORNER_MAPPING,
    VALID_BASES,
    FCGRMetadata,
    FCGRResult,
    compute_cgr_points,
    compute_fcgr,
    count_base_composition,
    count_kmers_in_run,
    sequence_checksum,
    split_into_valid_runs,
    uppercase_sequence,
    verify_kmer_mapping_bijectivity,
    _build_kmer_to_index,
)


# ═════════════════════════════════════════════════════════════════════════════
# Sequence utilities
# ═════════════════════════════════════════════════════════════════════════════


class TestUppercaseSequence:
    def test_lowercase(self):
        assert uppercase_sequence("acgt") == "ACGT"

    def test_mixed_case(self):
        assert uppercase_sequence("AcGtNn") == "ACGTNN"

    def test_empty(self):
        assert uppercase_sequence("") == ""


class TestSplitIntoValidRuns:
    def test_all_valid(self):
        assert split_into_valid_runs("ACGTACGT") == ["ACGTACGT"]

    def test_single_ambiguous(self):
        assert split_into_valid_runs("ACGTNACGT") == ["ACGT", "ACGT"]

    def test_multiple_ambiguous(self):
        assert split_into_valid_runs("ACNNGTTTNAC") == ["AC", "GTTT", "AC"]

    def test_leading_ambiguous(self):
        assert split_into_valid_runs("NNACGT") == ["ACGT"]

    def test_trailing_ambiguous(self):
        assert split_into_valid_runs("ACGTNN") == ["ACGT"]

    def test_all_ambiguous(self):
        assert split_into_valid_runs("NNNN") == []

    def test_empty(self):
        assert split_into_valid_runs("") == []

    def test_single_base(self):
        assert split_into_valid_runs("A") == ["A"]

    def test_iupac_codes(self):
        """R, Y, S, W, etc. are ambiguous and should split runs."""
        assert split_into_valid_runs("ACRYGT") == ["AC", "GT"]


class TestCountBaseComposition:
    def test_simple(self):
        result = count_base_composition("AACCGGTT")
        assert result["A"] == 2
        assert result["C"] == 2
        assert result["G"] == 2
        assert result["T"] == 2
        assert result["valid"] == 8
        assert result["ambiguous"] == 0
        assert result["total"] == 8

    def test_with_ambiguous(self):
        result = count_base_composition("ACNGT")
        assert result["valid"] == 4
        assert result["ambiguous"] == 1
        assert result["total"] == 5

    def test_empty(self):
        result = count_base_composition("")
        assert result["total"] == 0


class TestSequenceChecksum:
    def test_deterministic(self):
        seq = "ACGTACGT"
        assert sequence_checksum(seq) == sequence_checksum(seq)

    def test_matches_hashlib(self):
        seq = "ACGTACGT"
        expected = hashlib.sha256(seq.encode("ascii")).hexdigest()
        assert sequence_checksum(seq) == expected

    def test_different_sequences(self):
        assert sequence_checksum("ACGT") != sequence_checksum("TGCA")


# ═════════════════════════════════════════════════════════════════════════════
# k-mer to grid mapping bijectivity
# ═════════════════════════════════════════════════════════════════════════════


class TestKmerMappingBijectivity:
    """Verify the reversible index mapping test for supported small k.
    Required by projectContext.md §5.
    """

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    def test_bijectivity(self, k):
        """Every k-mer maps to a unique cell and the mapping covers the full grid."""
        assert verify_kmer_mapping_bijectivity(k) is True

    def test_k1_explicit(self):
        """At k=1, each nucleotide should map to its own quadrant."""
        mapping = _build_kmer_to_index(1, DEFAULT_CORNER_MAPPING)
        assert len(mapping) == 4
        # All four cells should be distinct
        cells = set(mapping.values())
        assert len(cells) == 4

    def test_k2_count(self):
        """At k=2, we should have 16 unique k-mers mapping to a 4x4 grid."""
        mapping = _build_kmer_to_index(2, DEFAULT_CORNER_MAPPING)
        assert len(mapping) == 16
        cells = set(mapping.values())
        assert len(cells) == 16

    def test_all_indices_in_bounds(self):
        """All indices should be within [0, 2^k)."""
        for k in range(1, 6):
            grid_size = 2**k
            mapping = _build_kmer_to_index(k, DEFAULT_CORNER_MAPPING)
            for kmer, (row, col) in mapping.items():
                assert 0 <= row < grid_size, f"k={k}, {kmer}: row {row} out of bounds"
                assert 0 <= col < grid_size, f"k={k}, {kmer}: col {col} out of bounds"


# ═════════════════════════════════════════════════════════════════════════════
# CGR point generation
# ═════════════════════════════════════════════════════════════════════════════


class TestComputeCGRPoints:
    def test_empty_sequence(self):
        points = compute_cgr_points("")
        assert points.shape == (0, 2)

    def test_all_ambiguous(self):
        points = compute_cgr_points("NNNN")
        assert points.shape == (0, 2)

    def test_single_base(self):
        """First point should be halfway between center (0.5, 0.5) and the corner."""
        points = compute_cgr_points("A")
        assert points.shape == (1, 2)
        # A corner = (0, 0); midpoint of (0.5, 0.5) and (0, 0) = (0.25, 0.25)
        np.testing.assert_allclose(points[0], [0.25, 0.25])

    def test_deterministic(self):
        """CGR must be deterministic (projectContext.md §4.3)."""
        seq = "ACGTACGTACGT"
        p1 = compute_cgr_points(seq)
        p2 = compute_cgr_points(seq)
        np.testing.assert_array_equal(p1, p2)

    def test_points_in_unit_square(self):
        """All CGR points should remain within [0, 1] × [0, 1]."""
        seq = "ACGTACGTACGTACGTACGTACGT"
        points = compute_cgr_points(seq)
        assert np.all(points >= 0.0)
        assert np.all(points <= 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# FCGR computation and invariants
# ═════════════════════════════════════════════════════════════════════════════


class TestComputeFCGR:
    """Test the primary FCGR builder and all invariants from projectContext.md §4.3."""

    def test_basic_k1(self):
        """k=1: each base contributes one 1-mer to its cell."""
        result = compute_fcgr("ACGT", k=1, normalization="raw")
        assert result.matrix.shape == (2, 2)
        assert result.raw_counts.sum() == 4  # 4 valid 1-mers
        assert result.metadata.valid_kmer_count == 4

    def test_shape(self):
        """Matrix shape must be (2^k, 2^k)."""
        for k in [1, 2, 3, 4]:
            result = compute_fcgr("ACGTACGTACGTACGT", k=k, normalization="raw")
            expected_size = 2**k
            assert result.matrix.shape == (expected_size, expected_size), f"k={k}"

    def test_invariant_sum_equals_valid_kmers(self):
        """sum(raw_matrix) == valid overlapping k-mer count."""
        seq = "ACGTACGTACGT"
        for k in [1, 2, 3, 4]:
            result = compute_fcgr(seq, k=k, normalization="raw")
            expected_count = len(seq) - k + 1
            assert int(result.raw_counts.sum()) == expected_count, f"k={k}"
            assert result.metadata.valid_kmer_count == expected_count, f"k={k}"

    def test_invariant_entries_nonnegative(self):
        """All matrix entries must be nonnegative."""
        result = compute_fcgr("ACGTACGTACGT", k=3, normalization="count")
        assert np.all(result.matrix >= 0)
        assert np.all(result.raw_counts >= 0)

    def test_invariant_entries_finite(self):
        """All matrix entries must be finite."""
        result = compute_fcgr("ACGTACGTACGT", k=3, normalization="count")
        assert np.all(np.isfinite(result.matrix))
        assert np.all(np.isfinite(result.raw_counts))

    def test_invariant_count_normalization_sums_to_one(self):
        """Count-normalized matrix must sum to 1.0."""
        result = compute_fcgr("ACGTACGTACGTACGTACGT", k=3, normalization="count")
        assert abs(float(result.matrix.sum()) - 1.0) < 1e-5

    def test_invariant_deterministic(self):
        """Output must be deterministic."""
        seq = "ACGTACGTACGTACGT"
        r1 = compute_fcgr(seq, k=3, normalization="count")
        r2 = compute_fcgr(seq, k=3, normalization="count")
        np.testing.assert_array_equal(r1.matrix, r2.matrix)
        np.testing.assert_array_equal(r1.raw_counts, r2.raw_counts)
        assert r1.metadata == r2.metadata

    def test_empty_sequence_raises(self):
        """Empty/short sequences must raise ValueError, not produce all-zero."""
        with pytest.raises(ValueError, match="no valid"):
            compute_fcgr("", k=3)

    def test_short_sequence_raises(self):
        """Sequence shorter than k must raise ValueError."""
        with pytest.raises(ValueError, match="no valid"):
            compute_fcgr("AC", k=3)

    def test_all_ambiguous_raises(self):
        """All-ambiguous sequence must raise ValueError."""
        with pytest.raises(ValueError, match="no valid"):
            compute_fcgr("NNNNNN", k=3)

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            compute_fcgr("ACGT", k=0)

    def test_invalid_normalization(self):
        with pytest.raises(ValueError, match="Unknown normalization"):
            compute_fcgr("ACGT", k=1, normalization="invalid")

    def test_invalid_ambiguity_policy(self):
        with pytest.raises(ValueError, match="Unknown ambiguity_policy"):
            compute_fcgr("ACGT", k=1, ambiguity_policy="ignore_all")


class TestFCGRAmbiguityHandling:
    """Test that ambiguous bases correctly split k-mer counting."""

    def test_no_cross_boundary_kmers(self):
        """k-mers must not span across ambiguous bases.

        "ACNGT" with k=2 should give: AC from "AC" run, GT from "GT" run.
        NOT: CN, NG which would cross the N boundary.
        """
        result = compute_fcgr("ACNGT", k=2, normalization="raw")
        # "AC" run has 1 valid 2-mer: "AC"
        # "GT" run has 1 valid 2-mer: "GT"
        assert result.metadata.valid_kmer_count == 2
        assert int(result.raw_counts.sum()) == 2

    def test_multiple_ambiguous_bases(self):
        """Multiple N's should not create phantom k-mers."""
        result = compute_fcgr("ACGTNNNNACGT", k=2, normalization="raw")
        # Two runs of "ACGT", each with 3 valid 2-mers
        assert result.metadata.valid_kmer_count == 6

    def test_run_too_short_for_k(self):
        """A valid run shorter than k should contribute zero k-mers."""
        result = compute_fcgr("ACNGNACGT", k=3, normalization="raw")
        # Runs: "AC" (too short for k=3), "G" (too short), "ACGT" (2 valid 3-mers)
        assert result.metadata.valid_kmer_count == 2

    def test_ambiguity_metadata(self):
        """Metadata should accurately reflect ambiguous base counts."""
        result = compute_fcgr("ACNGT", k=1)
        assert result.metadata.ambiguous_bases == 1
        assert result.metadata.valid_bases == 4
        assert result.metadata.total_bases == 5
        assert result.metadata.n_runs == 2


class TestFCGRNormalization:
    """Test different normalization modes."""

    def test_raw_normalization(self):
        """Raw mode should return integer counts as float32."""
        result = compute_fcgr("ACGTACGT", k=2, normalization="raw")
        # Counts should be whole numbers
        np.testing.assert_array_equal(result.matrix, result.matrix.astype(int))

    def test_count_normalization(self):
        """Count normalization divides by total valid k-mers."""
        result = compute_fcgr("ACGTACGT", k=2, normalization="count")
        assert abs(float(result.matrix.sum()) - 1.0) < 1e-5

    def test_log1p_normalization(self):
        """log1p normalization applies log(1 + count)."""
        result_raw = compute_fcgr("ACGTACGT", k=2, normalization="raw")
        result_log = compute_fcgr("ACGTACGT", k=2, normalization="log1p")
        expected = np.log1p(result_raw.raw_counts.astype(np.float32))
        np.testing.assert_allclose(result_log.matrix, expected, rtol=1e-5)


class TestFCGRMetadata:
    """Test that metadata is complete and correct."""

    def test_metadata_fields(self):
        result = compute_fcgr("ACGTACGT", k=3)
        m = result.metadata

        assert m.k == 3
        assert m.corner_mapping == DEFAULT_CORNER_MAPPING
        assert m.row_origin == "top"
        assert m.normalization == "count"
        assert m.ambiguity_policy == "split_runs"
        assert m.valid_kmer_count == 6  # 8 - 3 + 1
        assert m.total_bases == 8
        assert m.valid_bases == 8
        assert m.ambiguous_bases == 0
        assert m.n_runs == 1
        assert len(m.sequence_checksum) == 64  # SHA-256 hex
        assert m.implementation_version == "0.1.0"
        assert m.dtype == "float32"

    def test_checksum_in_metadata(self):
        """Metadata checksum should match independent computation."""
        seq = "ACGTACGT"
        result = compute_fcgr(seq, k=2)
        expected_checksum = hashlib.sha256(seq.encode("ascii")).hexdigest()
        assert result.metadata.sequence_checksum == expected_checksum


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases and stress tests
# ═════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_single_base_k1(self):
        """Single base with k=1 should work."""
        result = compute_fcgr("A", k=1, normalization="count")
        assert result.metadata.valid_kmer_count == 1
        assert abs(float(result.matrix.sum()) - 1.0) < 1e-5

    def test_homopolymer(self):
        """All same base — one cell gets all counts."""
        result = compute_fcgr("AAAAAAA", k=2, normalization="raw")
        assert int(result.raw_counts.sum()) == 6  # 7 - 2 + 1
        # Only the "AA" cell should have counts
        assert int(result.raw_counts.max()) == 6

    def test_lowercase_input(self):
        """Lowercase input should be handled (uppercased internally)."""
        r1 = compute_fcgr("acgtacgt", k=2)
        r2 = compute_fcgr("ACGTACGT", k=2)
        np.testing.assert_array_equal(r1.matrix, r2.matrix)

    def test_case_insensitive_checksum(self):
        """Checksum should be of the uppercased sequence."""
        r1 = compute_fcgr("acgt", k=1)
        r2 = compute_fcgr("ACGT", k=1)
        assert r1.metadata.sequence_checksum == r2.metadata.sequence_checksum


class TestCountKmersInRun:
    """Unit tests for the k-mer counting helper."""

    def test_exact_k_length(self):
        """Run of exactly k bases produces 1 k-mer."""
        grid_size = 4  # k=2
        mapping = _build_kmer_to_index(2, DEFAULT_CORNER_MAPPING)
        matrix = np.zeros((grid_size, grid_size), dtype=np.int64)
        count = count_kmers_in_run("AC", 2, mapping, matrix)
        assert count == 1
        assert int(matrix.sum()) == 1

    def test_shorter_than_k(self):
        """Run shorter than k produces 0 k-mers."""
        mapping = _build_kmer_to_index(3, DEFAULT_CORNER_MAPPING)
        matrix = np.zeros((8, 8), dtype=np.int64)
        count = count_kmers_in_run("AC", 3, mapping, matrix)
        assert count == 0
        assert int(matrix.sum()) == 0

    def test_empty_run(self):
        mapping = _build_kmer_to_index(2, DEFAULT_CORNER_MAPPING)
        matrix = np.zeros((4, 4), dtype=np.int64)
        count = count_kmers_in_run("", 2, mapping, matrix)
        assert count == 0
