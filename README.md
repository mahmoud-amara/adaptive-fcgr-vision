# Adaptive CGR-Vision

### Adaptive Alignment-Free Genomic Classification Using Image Processing and Machine Learning

An alignment-free genomic classification research framework that investigates whether adaptive image-processing feature fusion can improve bacterial genus classification using Frequency Chaos Game Representation (FCGR) and machine learning.

The framework transforms genomic DNA sequences into image-like representations, extracts complementary visual features, and evaluates adaptive feature fusion under a leakage-aware experimental protocol.

---

## 1. Research Architecture

```text
DNA Sequences (FASTA)
        |
        v
Quality Control and Validation
        |
        v
Bijective Chaos Game Mapping
        |
        v
FCGR Matrix (k=6, 64x64)
        |
        +-------------------+-------------------+
        |                   |                   |
        v                   v                   v
   GLCM Texture       Spatial Moments      Frequency Features
        |                   |                   |
        v                   v                   v
   Contrast             Centroid             2D DCT
   Homogeneity          Variances            Haar Wavelets
   Entropy              Quadrant Mass        Frequency Energy
   Energy               Spatial Entropy
        |                   |                   |
        +-------------------+-------------------+
                            |
                            v
                    Edge / Contrast Features
                            |
                            v
                    Feature Extraction
                            |
                            v
                Adaptive Feature Fusion
                (Fold-Local Weight Learning)
                            |
                            v
                  Machine Learning Models
                            |
                            v
                  Evaluation and Analysis
```

## 2. FCGR Representation

The framework uses Frequency Chaos Game Representation to convert DNA sequences into two-dimensional numerical matrices.

Primary representation:
- k-mer order: k = 6.
- Matrix dimensions: 64 × 64.
- Total matrix cells: 4,096.
- Bijective k-mer-to-grid mapping.
- Count-based FCGR representation.
- IUPAC ambiguity handling using sequence splitting.

## 3. Image Processing Feature Extraction

The framework implements four complementary feature families.

### Family A: GLCM Texture Features

Gray-Level Co-occurrence Matrix features capture local texture patterns.

Extracted properties:
- Contrast
- Dissimilarity
- Homogeneity
- Energy
- Correlation
- Entropy

Confirmatory configuration:
- 32 gray-level quantization levels.
- Log1p equal-width quantization.
- Distances: 1, 2, 3.
- Angles: 0°, 45°, 90°, 135°.
- Direction aggregation: mean.

### Family B: Spatial Features

Spatial features describe the distribution of FCGR intensity across the image.

Extracted properties:
- Spatial centroid.
- Row and column variance.
- Spatial covariance.
- Quadrant mass distribution.
- Spatial entropy.
- Row and column marginal statistics.

### Family C: Multiscale Frequency Features

Frequency-domain analysis captures global and multiscale patterns.

Methods:
- Two-dimensional Discrete Cosine Transform (DCT).
- DCT frequency-band energy.
- Low-frequency energy ratio.
- Haar wavelet decomposition.
- Wavelet subband energy.
- Wavelet energy distribution.

Confirmatory configuration:
- Four DCT frequency bands.
- Haar wavelet.
- Three decomposition levels.

### Family D: Edge and Contrast Features

Edge and local contrast analysis captures structural variations in FCGR images.

Methods:
- Sobel gradient magnitude.
- Edge magnitude statistics.
- Edge density at multiple thresholds.
- Local variance statistics.
- Global contrast range.
- Interquartile range (IQR).

---

## 4. Adaptive Feature Fusion

A central component of Adaptive CGR-Vision is adaptive feature fusion.

Instead of assigning identical importance to every feature family, the framework investigates optimized family-level weights.

The confirmatory configuration uses:

- Simplex grid search.
- Non-negative weights.
- Sum-to-one weight constraint.
- Inner cross-validation optimization.
- Macro-F1 as the optimization metric.

The fusion weights are selected within the training process to avoid using outer validation or test information.

A fixed feature fusion implementation is also included for comparison.

---

## 5. Experimental Dataset

The confirmatory configuration defines a balanced bacterial genome classification dataset.

| Parameter | Configuration |
|---|---|
| Data source | NCBI RefSeq |
| Classification level | Genus |
| Number of classes | 6 |
| Genomes per genus | 25 |
| Total genomes | 150 |
| Minimum genome length | 1 Mbp |
| Maximum ambiguity fraction | 1% |
| Primary replicon | Primary chromosome |

### Organism Panel

1. Escherichia
2. Salmonella
3. Pseudomonas
4. Bacillus
5. Staphylococcus
6. Mycobacterium

Assembly inclusion is restricted to Complete Genome and Chromosome levels.

Plasmids are excluded from the primary chromosome analysis.

---

## 6. Validation and Leakage Prevention

The confirmatory experimental protocol uses nested grouped cross-validation.

| Component | Configuration |
|---|---|
| Outer cross-validation | 5 folds |
| Inner cross-validation | 4 folds |
| Grouping method | MinHash / ANI-based grouping |
| Similarity threshold | 95% |
| k-mer size | 21 |
| Sketch size | 1,000 |

The grouping strategy is designed to reduce similarity-related information leakage between training and evaluation partitions.

All adaptive fusion optimization is performed within the training process.

---

## 7. Machine Learning Models and Baselines

The confirmatory configuration includes:

### Machine Learning Models
- Support Vector Machine (RBF kernel).
- Support Vector Machine (Linear kernel).
- Random Forest.

### Alignment-Based Baselines
- Smith-Waterman 1-NN on 16S rRNA sequences.
- Whole-genome FastANI baseline.

These methods provide reference points for evaluating the proposed alignment-free approach.

---

## 8. Evaluation Metrics and Statistical Analysis

The primary evaluation metric is Macro-F1.

Secondary metrics include:
- Accuracy.
- Balanced Accuracy.
- Expected Calibration Error (ECE).
- Brier Score.

Statistical evaluation includes:
- Bootstrap confidence intervals.
- Paired permutation testing.
- 1,000 bootstrap samples.
- 1,000 paired permutations.
- 95% confidence level.

---

## 9. Repository Structure

```text
.
├── configs/
│   ├── confirmatory.yaml
│   └── development.yaml
│
├── data/
│   └── manifests/
│
├── docs/
│   ├── decisions.md
│   └── feature_registry.md
│
├── scripts/
│   ├── acquire_dataset.py
│   ├── generate_publication_figures.py
│   ├── run_ablation_study.py
│   ├── run_computational_cost_benchmark.py
│   ├── run_final_research_package.py
│   ├── run_full_benchmark.py
│   └── run_staged_acquisition_and_pilot.py
│
├── src/
│   └── adaptive_cgr_vision/
│       ├── data/
│       ├── representations/
│       ├── features/
│       ├── fusion/
│       ├── models/
│       ├── evaluation/
│       ├── artifacts/
│       └── cli.py
│
├── tests/
│   ├── test_clustering.py
│   ├── test_fcgr.py
│   ├── test_features.py
│   └── test_pipeline.py
│
├── results/
├── reports/
├── pyproject.toml
├── .gitignore
├── LICENSE
└── README.md
```

---

## 10. Installation

### Requirements

- Python 3.11 or newer.
- pip.

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Install the project:

```bash
pip install -e .
```

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Optional dependencies:

```bash
pip install -e ".[sw]"
pip install -e ".[boost]"
```

---

## 11. Running Tests

Run the test suite:

```bash
python -m pytest
```

The tests cover:
- Clustering.
- FCGR representation.
- Image feature extraction.
- Pipeline behavior.

---

## 12. Running Research Experiments

Research experiment scripts are available in the `scripts/` directory.

Main entry points:

| Script | Purpose |
|---|---|
| `acquire_dataset.py` | Dataset acquisition |
| `run_staged_acquisition_and_pilot.py` | Staged acquisition and pilot experiments |
| `run_full_benchmark.py` | Full benchmark execution |
| `run_final_research_package.py` | Final research package execution |
| `run_ablation_study.py` | Ablation experiments |
| `run_computational_cost_benchmark.py` | Computational cost evaluation |
| `generate_publication_figures.py` | Publication figure generation |

Experimental parameters are maintained in `configs/`.

Check each script's available arguments and configuration requirements before execution.

---

## 13. Data Availability and Reproducibility

Raw genomic sequences and generated intermediate datasets are not included in this repository.

The following directories are excluded from version control:

- `data/raw/`
- `data/interim/`
- `data/processed/`
- `runs/`

Dataset manifests and methodological documentation are maintained in the repository where applicable.

Research decisions are documented in `docs/decisions.md`.

Feature definitions and parameters are documented in `docs/feature_registry.md`.

---

## 14. Project Status

Adaptive CGR-Vision is a research implementation and experimental framework.

Full experimental execution is performed through the research scripts in `scripts/`.

The command-line interface provides configuration and verification utilities; it is not currently presented as a complete automated training and evaluation pipeline.

Experimental results and publication artifacts are maintained separately from the source code.

---

## 15. License

This project is distributed under the MIT License.

See the `LICENSE` file for details.

---

## 16. Authors

- Mahmoud Amara
- Samaa Elfarghaly
- Jana Sherif
- Hend Mousa

---

**Adaptive CGR-Vision — Research Implementation**
