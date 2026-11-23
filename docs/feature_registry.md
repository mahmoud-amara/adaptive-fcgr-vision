# Feature Registry — Adaptive CGR-Vision

Version: 0.1.0-dev (PROVISIONAL — not approved for confirmatory runs)

This registry documents every image-processing feature extracted from FCGR matrices.
Each feature has a stable name, family prefix, formula, parameters, dtype, and
implementation reference.

> **Rule:** Do not alter this registry without human approval. Exploratory features
> must be clearly separated from confirmatory ones (AgentRules §4.5).

---

## Naming Convention

Feature names follow the pattern: `{family}_{descriptor}_{parameters}`

Examples:
- `glcm_contrast_d1_a0` — GLCM contrast at distance=1, angle=0°
- `spatial_mass_q00` — spatial mass in quadrant (0,0)
- `dct_energy_s1` — DCT energy at scale 1
- `edge_density_s2` — edge density at scale 2

---

## Family A: Texture (GLCM / Haralick)

| Feature Name | Formula | Parameters | Dtype | Notes |
|-------------|---------|------------|-------|-------|
| `glcm_contrast_d{d}_a{a}` | Σ (i-j)² · P(i,j) | distance d, angle a | float64 | Symmetric GLCM |
| `glcm_dissimilarity_d{d}_a{a}` | Σ |i-j| · P(i,j) | distance d, angle a | float64 | |
| `glcm_homogeneity_d{d}_a{a}` | Σ P(i,j)/(1+(i-j)²) | distance d, angle a | float64 | |
| `glcm_energy_d{d}_a{a}` | Σ P(i,j)² | distance d, angle a | float64 | Also called ASM |
| `glcm_correlation_d{d}_a{a}` | Σ (i-μᵢ)(j-μⱼ)P(i,j)/(σᵢσⱼ) | distance d, angle a | float64 | Zero-denominator → 0.0 |
| `glcm_entropy_d{d}_a{a}` | −Σ P(i,j)·log₂(P(i,j)) | distance d, angle a | float64 | 0·log(0) = 0 |

**Provisional parameters:**
- Quantization: log1p transform → 32 equal-width bins (PENDING PD-09)
- Distances: [1, 2, 3]
- Angles: [0°, 45°, 90°, 135°]
- Direction aggregation: mean across angles (produces per-distance features)
- Constant-image behavior: all features = 0.0

---

## Family B: Spatial Distribution and Moments

| Feature Name | Formula | Parameters | Dtype | Notes |
|-------------|---------|------------|-------|-------|
| `spatial_centroid_row` | Σ r·M(r,c) / Σ M(r,c) | Normalized FCGR | float64 | Center of mass row |
| `spatial_centroid_col` | Σ c·M(r,c) / Σ M(r,c) | Normalized FCGR | float64 | Center of mass col |
| `spatial_var_row` | Σ (r-μr)²·M(r,c) / Σ M | | float64 | Row variance |
| `spatial_var_col` | Σ (c-μc)²·M(r,c) / Σ M | | float64 | Col variance |
| `spatial_covar` | Σ (r-μr)(c-μc)·M(r,c) / Σ M | | float64 | Row-col covariance |
| `spatial_mass_q{rr}{cc}` | Σ M in quadrant | 4 quadrants | float64 | Normalized by total |
| `spatial_entropy` | −Σ p·log₂(p) | p = M/Σ M | float64 | 0·log(0) = 0 |
| `spatial_row_marginal_std` | std(Σ_c M(r,c)) | | float64 | Row marginal spread |
| `spatial_col_marginal_std` | std(Σ_r M(r,c)) | | float64 | Col marginal spread |

**Zero-mass behavior:** If total mass = 0, record is invalid (excluded by QC).

---

## Family C: Multiscale Frequency Summaries

| Feature Name | Formula | Parameters | Dtype | Notes |
|-------------|---------|------------|-------|-------|
| `dct_energy_s{s}` | Σ C(u,v)² in band s | s=0..N bands | float64 | 2D DCT, type-II |
| `dct_ratio_low` | E_low / E_total | | float64 | Low-freq concentration |
| `wavelet_energy_L{l}_{d}` | Σ coeff² in subband | level l, detail d | float64 | Haar/db4 wavelet |
| `wavelet_entropy_L{l}` | −Σ p·log₂(p) per level | | float64 | Energy distribution |

**Provisional parameters:**
- DCT: scipy.fft.dctn, type-II, ortho normalization; 4 frequency bands
- Wavelet: PyWavelets, Haar wavelet, 3 decomposition levels
- Coefficient aggregation: sum of squared coefficients per band/subband

---

## Family D: Local Contrast / Edge Statistics

| Feature Name | Formula | Parameters | Dtype | Notes |
|-------------|---------|------------|-------|-------|
| `edge_magnitude_mean` | mean(|∇M|) | Sobel gradient | float64 | |
| `edge_magnitude_std` | std(|∇M|) | Sobel gradient | float64 | |
| `edge_density_t{t}` | fraction(|∇M| > t) | threshold t | float64 | Fixed thresholds |
| `local_var_mean_s{s}` | mean of local variance | window s×s | float64 | |
| `local_var_std_s{s}` | std of local variance | window s×s | float64 | |
| `contrast_range` | max(M) - min(M) | | float64 | Global range |
| `contrast_iqr` | Q3(M) - Q1(M) | | float64 | Interquartile range |

**Provisional parameters:**
- Edge thresholds: [0.01, 0.05, 0.1] (relative to max gradient)
- Local variance window: [3, 5]
- All thresholds are fixed and versioned, not data-adaptive

---

## Registry Metadata

```yaml
registry_version: "0.1.0-dev"
status: "PROVISIONAL"
total_features_approx: ~55  # exact count depends on parameter grid
families: [glcm, spatial, dct_wavelet, edge_contrast]
fcgr_input: "count-normalized float32 FCGR matrix"
requires_approval: true
```
