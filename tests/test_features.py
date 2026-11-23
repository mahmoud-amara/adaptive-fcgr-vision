"""Unit and invariant tests for image-processing feature extraction modules.

Tests:
- Family A: GLCM / Haralick texture features
- Family B: Spatial distribution and moments
- Family C: Multiscale frequency (DCT and Wavelet)
- Family D: Edge and local contrast
- FeatureExtractor & Registry integration
- Invariants: finite values, determinism, shape handling, zero/constant matrix safety.
"""

import numpy as np
import pytest

from adaptive_cgr_vision.features.glcm import (
    quantize_matrix,
    compute_glcm_entropy,
    extract_glcm_features,
)
from adaptive_cgr_vision.features.spatial import extract_spatial_features
from adaptive_cgr_vision.features.frequency import (
    extract_dct_features,
    extract_wavelet_features,
    extract_frequency_features,
)
from adaptive_cgr_vision.features.edge import extract_edge_features
from adaptive_cgr_vision.features.extractor import FeatureExtractor
from adaptive_cgr_vision.features.registry import (
    get_feature_family,
    group_features_by_family,
)


@pytest.fixture
def sample_fcgr():
    """Create a non-trivial 64x64 pseudo-FCGR matrix."""
    rng = np.random.default_rng(42)
    mat = rng.exponential(scale=2.0, size=(64, 64)).astype(np.float64)
    # Add structured patterns
    mat[:32, :32] += 5.0
    mat[32:, 32:] += 10.0
    # Normalize to sum to 1
    return mat / np.sum(mat)


@pytest.fixture
def uniform_fcgr():
    """Uniform 64x64 matrix."""
    return np.ones((64, 64), dtype=np.float64) / (64 * 64)


@pytest.fixture
def zero_fcgr():
    """All-zero 64x64 matrix."""
    return np.zeros((64, 64), dtype=np.float64)


# ─── Family A: GLCM ─────────────────────────────────────────────────

class TestGLCMFeatures:
    def test_quantization_levels(self, sample_fcgr):
        q = quantize_matrix(sample_fcgr, n_levels=16, apply_log1p=True)
        assert q.shape == sample_fcgr.shape
        assert q.min() >= 0
        assert q.max() < 16
        assert q.dtype == np.uint8

    def test_quantization_constant_matrix(self, uniform_fcgr):
        q = quantize_matrix(uniform_fcgr, n_levels=32)
        assert np.all(q == 0)

    def test_glcm_extraction_keys(self, sample_fcgr):
        feats = extract_glcm_features(sample_fcgr, distances=[1, 2], aggregate_angles=True)
        expected_props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "entropy"]
        for prop in expected_props:
            for d in [1, 2]:
                assert f"glcm_{prop}_d{d}" in feats

    def test_glcm_finite_and_deterministic(self, sample_fcgr):
        f1 = extract_glcm_features(sample_fcgr)
        f2 = extract_glcm_features(sample_fcgr)
        for k, v in f1.items():
            assert np.isfinite(v), f"Feature {k} is not finite: {v}"
            assert v == f2[k]

    def test_glcm_zero_matrix(self, zero_fcgr):
        feats = extract_glcm_features(zero_fcgr)
        for k, v in feats.items():
            assert v == 0.0


# ─── Family B: Spatial Distribution ─────────────────────────────────

class TestSpatialFeatures:
    def test_spatial_quadrant_mass_sum(self, sample_fcgr):
        feats = extract_spatial_features(sample_fcgr)
        q_sum = (
            feats["spatial_mass_q00"]
            + feats["spatial_mass_q01"]
            + feats["spatial_mass_q10"]
            + feats["spatial_mass_q11"]
        )
        assert np.isclose(q_sum, 1.0, atol=1e-5)

    def test_spatial_centroid_in_bounds(self, sample_fcgr):
        feats = extract_spatial_features(sample_fcgr)
        assert 0.0 <= feats["spatial_centroid_row"] <= 63.0
        assert 0.0 <= feats["spatial_centroid_col"] <= 63.0

    def test_spatial_variances_nonnegative(self, sample_fcgr):
        feats = extract_spatial_features(sample_fcgr)
        assert feats["spatial_var_row"] >= 0.0
        assert feats["spatial_var_col"] >= 0.0
        assert feats["spatial_entropy"] >= 0.0
        assert feats["spatial_row_marginal_std"] >= 0.0
        assert feats["spatial_col_marginal_std"] >= 0.0

    def test_spatial_uniform_matrix_symmetry(self, uniform_fcgr):
        feats = extract_spatial_features(uniform_fcgr)
        # For 64x64 uniform, centroid should be exactly at center (31.5, 31.5)
        assert np.isclose(feats["spatial_centroid_row"], 31.5)
        assert np.isclose(feats["spatial_centroid_col"], 31.5)
        # Quadrants should each have exactly 0.25 mass
        assert np.isclose(feats["spatial_mass_q00"], 0.25)
        assert np.isclose(feats["spatial_mass_q01"], 0.25)
        assert np.isclose(feats["spatial_mass_q10"], 0.25)
        assert np.isclose(feats["spatial_mass_q11"], 0.25)


# ─── Family C: Multiscale Frequency ─────────────────────────────────

class TestFrequencyFeatures:
    def test_dct_energy_and_ratio(self, sample_fcgr):
        feats = extract_dct_features(sample_fcgr, n_bands=4)
        for s in range(4):
            assert f"dct_energy_s{s}" in feats
            assert feats[f"dct_energy_s{s}"] >= 0.0
        assert "dct_ratio_low" in feats
        assert 0.0 <= feats["dct_ratio_low"] <= 1.0

    def test_wavelet_energy_and_entropy(self, sample_fcgr):
        feats = extract_wavelet_features(sample_fcgr, wavelet="haar", max_levels=3)
        assert "wavelet_energy_approx" in feats
        for lvl in [1, 2, 3]:
            assert f"wavelet_energy_L{lvl}_H" in feats
            assert f"wavelet_energy_L{lvl}_V" in feats
            assert f"wavelet_energy_L{lvl}_D" in feats
            assert f"wavelet_entropy_L{lvl}" in feats
            assert feats[f"wavelet_entropy_L{lvl}"] >= 0.0

    def test_frequency_combined(self, sample_fcgr):
        combined = extract_frequency_features(sample_fcgr)
        assert "dct_ratio_low" in combined
        assert "wavelet_energy_approx" in combined
        for v in combined.values():
            assert np.isfinite(v)


# ─── Family D: Local Contrast & Edge ────────────────────────────────

class TestEdgeFeatures:
    def test_edge_features_keys(self, sample_fcgr):
        feats = extract_edge_features(sample_fcgr, thresholds=[0.01, 0.05], windows=[3, 5])
        assert "edge_magnitude_mean" in feats
        assert "edge_magnitude_std" in feats
        assert "contrast_range" in feats
        assert "contrast_iqr" in feats
        assert "edge_density_t001" in feats
        assert "edge_density_t005" in feats
        assert "local_var_mean_s3" in feats
        assert "local_var_std_s3" in feats
        assert "local_var_mean_s5" in feats
        assert "local_var_std_s5" in feats

    def test_edge_invariants(self, sample_fcgr):
        feats = extract_edge_features(sample_fcgr)
        assert feats["contrast_range"] >= 0.0
        assert feats["contrast_iqr"] >= 0.0
        assert feats["edge_magnitude_mean"] >= 0.0
        assert feats["edge_magnitude_std"] >= 0.0
        for k, v in feats.items():
            assert np.isfinite(v)


# ─── Extractor & Registry Integration ───────────────────────────────

class TestFeatureExtractorIntegration:
    def test_unified_extractor_shape(self, sample_fcgr):
        extractor = FeatureExtractor()
        feats = extractor.extract_from_matrix(sample_fcgr)
        # Should extract ~40-60 features across all 4 families
        assert len(feats) > 30
        assert all(np.isfinite(v) for v in feats.values())

    def test_extractor_deterministic_order(self, sample_fcgr):
        extractor = FeatureExtractor()
        f1 = extractor.extract_from_matrix(sample_fcgr)
        f2 = extractor.extract_from_matrix(sample_fcgr)
        assert list(f1.keys()) == list(f2.keys())
        assert list(f1.values()) == list(f2.values())

    def test_family_grouping(self, sample_fcgr):
        extractor = FeatureExtractor()
        feats = extractor.extract_from_matrix(sample_fcgr)
        grouped = group_features_by_family(list(feats.keys()))
        assert len(grouped["glcm"]) > 0
        assert len(grouped["spatial"]) > 0
        assert len(grouped["frequency"]) > 0
        assert len(grouped["edge"]) > 0
        total_grouped = sum(len(v) for v in grouped.values())
        assert total_grouped == len(feats)

    def test_extract_from_sequence(self):
        extractor = FeatureExtractor()
        seq = "ACGTACGTNNGTTTGACCGT" * 50
        feats = extractor.extract_from_sequence(seq, k=4)
        assert len(feats) > 0
        assert all(np.isfinite(v) for v in feats.values())
