# Final Computational Cost Analysis & Publication Figure Overhaul Audit Report

**Project**: Adaptive CGR-Vision: Multiscale Computer Vision Descriptors for Genomic Classification  
**Date**: 2026-10-12  
**Audit Scope**: Computational cost benchmarking, memory/throughput profiling, publication-quality figure redesign suite, manuscript integration, and reproducibility audit.  
**Location**: [`runs/final_research_package/computational_cost_and_figure_audit.md`](file:///d:/Image%20Research%20paper/runs/final_research_package/computational_cost_and_figure_audit.md)  
**Status**: Completed & Verified

---

## 1. What Was Inspected & Verified

Prior to implementing any changes, the following project components and historical artifacts were inspected:
1. **Governing Guidelines**: Verified `AgentRules.md` and `projectContext.md` to ensure adherence to zero-leakage, non-fabrication of citations/metrics, and preservation of raw data/splits.
2. **Saved Benchmark Data**: Re-verified pooled Macro-F1 and per-fold metrics in:
   - [`runs/final_research_package/final_benchmark_results.json`](file:///d:/Image%20Research%20paper/runs/final_research_package/final_benchmark_results.json)
   - [`runs/ablation_study/ablation_results.json`](file:///d:/Image%20Research%20paper/runs/ablation_study/ablation_results.json)
   - [`runs/full_benchmark/metrics.json`](file:///d:/Image%20Research%20paper/runs/full_benchmark/metrics.json)
3. **Existing Figures & Scripts**: Audited the four original figures in [`runs/final_research_package/figures/`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures) and created timestamped backups before modification.
4. **Data Manifests & Cache**: Confirmed integrity of:
   - `data/manifests/dataset_manifest_full.csv` (150 RefSeq bacterial accessions)
   - `data/manifests/splits_full.csv` (5-fold outer grouped splits, 149 clusters, zero inter-fold leakage)
   - `data/processed/fcgr_matrices_cache.pkl` (150 precomputed $64 \times 64$ matrices)

---

## 2. Model Selection Rationale

As required by Task 1.1, the computational benchmark selected the two highest-performing verified models and established a matched-classifier comparison isolating representation effects:

1. **`Raw_FCGR_4096D__SVM_RBF`** (Rank 1 Overall Performance):
   - *Pooled Macro-F1*: **0.9471**, *Accuracy*: **94.67%**.
   - *Rationale*: Reference baseline for the raw 4,096-pixel representation using an RBF kernel machine ($C=1.0, \gamma=\text{"scale"}$).
2. **`Raw_FCGR_4096D__RandomForest`** (Rank 2 Overall Performance):
   - *Pooled Macro-F1*: **0.9336**, *Accuracy*: **93.33%**.
   - *Rationale*: Reference baseline for the raw 4,096-pixel representation using a decision tree ensemble (100 trees).
3. **`Multiscale_59D__SVM_RBF`** (Matched Comparison to Rank 1):
   - *Pooled Macro-F1*: **0.9200**, *Accuracy*: **92.00%**.
   - *Rationale*: Directly isolates the effect of representation ($4,096\text{D} \to 59\text{D}$) while holding the classifier architecture constant (identical SVM-RBF parameters and random seed).
4. **`Multiscale_59D__RandomForest`** (Matched Comparison to Rank 2 & Primary Proposed Model):
   - *Pooled Macro-F1*: **0.9249**, *Accuracy*: **92.67%**.
   - *Rationale*: Directly evaluates representation compression on Random Forest, controlling for tree ensemble parameters.
5. **`GLCM_18D__RandomForest`** (Highest-Performing Single Family Baseline):
   - *Pooled Macro-F1*: **0.9329**, *Accuracy*: **93.33%**.
   - *Rationale*: Benchmarks whether single high-capacity visual textures offer competitive runtime efficiency.
6. **`Mash_1NN__Distance_Baseline`** (Reference Alignment-Free Benchmark):
   - *Pooled Macro-F1*: **0.8814**, *Accuracy*: **87.33%**.
   - *Rationale*: Evaluates computational cost and latency of traditional MinHash sequence sketching ($k=21, s=1000$).

---

## 3. Hardware & Measurement Protocol

- **Hardware Platform**: 13th Gen Intel(R) Core(TM) i7-13650HX (14 physical cores, 20 logical threads).
- **Installed RAM**: 16.0 GB DDR5 RAM (15.71 GB visible).
- **Operating System**: Windows 11 Pro (build 10.0.26300).
- **Python Runtime**: Python 3.14.3 (tags/v3.14.3:323c59a, 64-bit AMD64).
- **Key Dependencies**: NumPy 2.4.4, Scikit-Learn 1.9.1, SciPy 1.18.1, Matplotlib 3.10.8.
- **Timing Disciplines**:
  - `time.perf_counter()` high-precision CPU timers.
  - Warm-up cycles executed prior to all timing loops to eliminate cold-start cache misses.
  - Repeated measurements across 5 outer CV folds (3–20 repetitions per fold).
  - Peak memory allocated tracked via Python standard library `tracemalloc`.
  - Zero leakage: `StandardScaler` fitted strictly on training folds ($N=120$) and applied to test folds ($N=30$).

---

## 4. Empirical Computational Cost & Efficiency Results

### 4.1 Feature Matrix Dimensionality & Storage Footprint
Feature array shapes and memory footprints for all 150 bacterial genomes ($N=150$, `float64` numeric dtype):

| Representation | Vector Dimension ($D$) | Numeric Dtype | Bytes / Genome | 150-Genome Matrix Size | Compression vs. Raw 4096D |
|---|---|---|---|---|---|
| **Raw FCGR Grid** | 4,096 | `float64` | 32,768 B (32 KB) | **4,800.00 KB (4.69 MB)** | 1.00× (Baseline) |
| **Multiscale 59D (Ours)** | **59** | `float64` | **472 B (0.46 KB)** | **69.14 KB (0.07 MB)** | **69.42× Reduction** |
| **GLCM Texture** | 18 | `float64` | 144 B | 21.09 KB | 227.56× Reduction |
| **Spatial Moments** | 12 | `float64` | 96 B | 14.06 KB | 341.33× Reduction |
| **Frequency Spectrum** | 18 | `float64` | 144 B | 21.09 KB | 227.56× Reduction |
| **Edge & Boundary** | 11 | `float64` | 88 B | 12.89 KB | 372.36× Reduction |
| **Mash MinHash Sketch** | 1,000 | `uint64` | 8,000 B | 1,171.88 KB (1.14 MB) | 4.10× Reduction |

### 4.2 Sequence-to-FCGR Matrix Construction Time
Measured across 6 representative RefSeq chromosomes (lengths $1.35\text{--}6.26\text{ Mbp}$):
- **Mean time per genome**: **$4.486 \pm 1.832\text{ seconds}$**
- **Median time per genome**: **$3.407\text{ seconds}$**
- **Throughput per megabase**: **$0.990 \pm 0.321\text{ s/Mbp}$** (approx. $1.01\text{ Mbp/s}$ single-thread counting).

### 4.3 Feature Extraction from Precomputed FCGR Grids
Measured across 5 repetitions of all 150 genomes:
- **Multiscale 59D**: **$8.053 \pm 0.327\text{ ms per genome}$** (median: **$8.059\text{ ms}$**, throughput: **$124.2\text{ genomes/s}$**, peak RAM: **$469.1\text{ KB}$**).
- **GLCM Texture (18D)**: **$3.147 \pm 0.039\text{ ms per genome}$** (median: **$3.141\text{ ms}$**, throughput: **$317.8\text{ genomes/s}$**, peak RAM: **$431.8\text{ KB}$**).
- **Raw FCGR (Flatten)**: **$0.003\text{ ms per genome}$** (peak RAM: **$32.7\text{ KB}$**).

### 4.4 Model Training & Inference Benchmarks (5 Outer Folds)
All metrics measured across 5 outer folds ($N_{\text{train}}=120, N_{\text{test}}=30$):

| Model Configuration | Feature Dim | Test Macro-F1 (Mean ± SD) | Train Time (Mean ms) | Single Latency (Median ms) | Batch-30 Throughput (Genomes/s) | Peak Heap Alloc. (KB) |
|---|---|---|---|---|---|---|
| **Raw FCGR + SVM-RBF** (Top 1) | 4,096 | **0.9472 ± 0.0334** | 17.94 ± 1.56 | 0.327 | 5,640.1 | 3,850.2 |
| **Multiscale 59D + SVM-RBF** (Matched) | 59 | **0.9195 ± 0.0340** | **5.01 ± 0.12** | **0.129** | **80,578.2** | **63.2** |
| **Raw FCGR + RandomForest** (Top 2) | 4,096 | **0.9332 ± 0.0204** | 846.41 ± 11.06 | 5.997 | 4,617.0 | 2,165.7 |
| **Multiscale 59D + RF** (Proposed) | 59 | **0.9246 ± 0.0267** | 822.05 ± 330.57 | 11.170 | 2,268.1 | **205.9** |
| **Multiscale 59D + RF (Regularized)** | 59 | 0.9122 ± 0.0339 | 1,758.69 ± 709.46 | 26.885 | 1,244.1 | 205.2 |
| **GLCM 18D + RandomForest** | 18 | 0.9331 ± 0.0371 | 2,563.42 ± 101.88 | 26.992 | 1,039.7 | 185.3 |
| **Mash 1-NN Distance Baseline** | 1,000 | 0.8770 ± 0.0373 | 0.31 ± 0.13 | 80.854 | 11.9 | 11.2 |

#### Computational Comparison Highlights:
- **Matched SVM-RBF Latency Speedup**: Reducing dimensionality from 4,096 to 59 lowers median single-genome inference latency from $0.327\text{ ms}$ to $0.129\text{ ms}$ (**$2.53\times$ speedup**).
- **Matched SVM-RBF Batch Throughput**: Vectorized matrix inference throughput accelerates from $5,640\text{ genomes/s}$ to $80,578\text{ genomes/s}$ (**$14.29\times$ higher throughput**).
- **Matched SVM-RBF Training Speedup**: Training time drops from $17.94\text{ ms}$ to $5.01\text{ ms}$ (**$3.58\times$ training speedup**).
- **Dynamic Training Heap Reduction**: Peak heap allocation during SVM-RBF training drops from $3,850.2\text{ KB}$ to $63.2\text{ KB}$ (**$60.9\times$ reduction**).
- **Random Forest Dynamic Heap**: Peak heap allocation during Random Forest training drops from $2,165.7\text{ KB}$ to $205.9\text{ KB}$ (**$10.5\times$ reduction**).

---

## 5. Figure Overhaul & Artifacts Inventory

All figures were created or redesigned with a consistent, publication-ready style:
- High-resolution 300 DPI PNG and vector PDF outputs.
- Colorblind-friendly palette (Okabe-Ito / Paul Tol / ColorBrewer).
- Clean sans-serif typography, no overlapping text, clear axis limits and annotations.

### 5.1 Redesigned Existing Figures (Task 3)
1. [`fig1_model_performance_bar`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig1_model_performance_bar.png) ([PDF](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig1_model_performance_bar.pdf)): Horizontal bar chart comparing all 10 models with exact F1 and accuracy values.
2. [`fig2_ablation_family_f1`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig2_ablation_family_f1.png) ([PDF](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig2_ablation_family_f1.pdf)): Grouped bar chart comparing SVM-RBF and Random Forest across single families and the combined 59D representation.
3. [`fig3_confusion_matrices`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig3_confusion_matrices.png) ([PDF](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig3_confusion_matrices.pdf)): Out-of-fold confusion matrix for *Multiscale 59D + RF* with italicized genus names, integer counts, and recall percentages.
4. [`fig4_family_importance_distribution`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig4_family_importance_distribution.png) ([PDF](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig4_family_importance_distribution.pdf)): Horizontal ranked contribution chart of ANOVA F-score family weights.

### 5.2 Newly Created Figures (Task 2)
5. [`fig_a_performance_vs_cost`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig_a_performance_vs_cost.png) ([PDF](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig_a_performance_vs_cost.pdf)): Pareto trade-off scatter plot comparing Macro-F1 vs. single-genome inference latency (log scale).
6. [`fig_b_stage_by_stage_runtime`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig_b_stage_by_stage_runtime.png) ([PDF](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig_b_stage_by_stage_runtime.pdf)): Dual-panel chart separating upstream sequence-to-FCGR generation (seconds scale) from downstream feature extraction and inference (ms scale).
7. [`fig_c_memory_and_storage_footprint`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig_c_memory_and_storage_footprint.png) ([PDF](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig_c_memory_and_storage_footprint.pdf)): Dual-panel memory analysis comparing feature matrix storage ($69.4\times$ reduction) and peak training process RAM ($60.9\times$ reduction).
8. [`fig_d_performance_cost_tradeoff`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig_d_performance_cost_tradeoff.png) ([PDF](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig_d_performance_cost_tradeoff.pdf)): 4-panel multi-criteria trade-off dashboard: Quality, Storage, Training Time, and Batch Throughput.
9. [`fig_e_pipeline_schematic`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig_e_pipeline_schematic.png) ([PDF](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/fig_e_pipeline_schematic.pdf)): Complete end-to-end scientific architecture schematic.

---

## 6. Verification Checklist Status

| Item | Requirement | Status | Verification Evidence |
|---|---|---|---|
| **1** | Project structure & existing instructions preserved | ✅ **PASSED** | Repository layout, manifests, scripts, and source modules intact. |
| **2** | Original predictions, manifests, splits unchanged | ✅ **PASSED** | Checked SHA-256 and verified no modifications to `data/manifests/` or prior prediction arrays. |
| **3** | Cost measurements reproducible from saved scripts | ✅ **PASSED** | Script [`scripts/run_computational_cost_benchmark.py`](file:///d:/Image%20Research%20paper/scripts/run_computational_cost_benchmark.py) executes deterministically. |
| **4** | Distinct timing across pipeline stages | ✅ **PASSED** | Sequence-to-FCGR, feature extraction, training, and inference timed independently. |
| **5** | Selected models & selection criteria documented | ✅ **PASSED** | Documented in Section 2 above and in the cost report. |
| **6** | All figures use verified data | ✅ **PASSED** | Cross-referenced against `final_benchmark_results.json` and `computational_cost_results.json`. |
| **7** | Existing figures inventoried and reviewed | ✅ **PASSED** | Detailed in [`figure_inventory.md`](file:///d:/Image%20Research%20paper/runs/final_research_package/figure_inventory.md). |
| **8** | No fabricated measurements or unverified claims | ✅ **PASSED** | All numbers are directly computed from test executions; zero synthetic values. |
| **9** | Every figure has caption and documented source | ✅ **PASSED** | Full captions and descriptions provided in [`figure_change_log.md`](file:///d:/Image%20Research%20paper/runs/final_research_package/figure_change_log.md). |
| **10** | Revised manuscript agrees with numerical results | ✅ **PASSED** | Updated Sections 4.5 and 5.1 in [`adaptive_cgr_vision_research_draft.md`](file:///d:/Image%20Research%20paper/runs/final_research_package/adaptive_cgr_vision_research_draft.md). |
| **11** | All backups and output locations documented | ✅ **PASSED** | Backups saved in `runs/final_research_package/figures/backup_20261012_164106/` and `runs/final_research_package/adaptive_cgr_vision_research_draft_backup_20261012.md`. |

---

## 7. Remaining Methodological Limitations

1. **Closed-Set Taxonomic Evaluation**: The 150-genome panel covers 6 bacterial genera. Generalization to unseen bacterial phyla, divergent strains, or fragmented metagenomic contigs requires future evaluation.
2. **Upstream Single-Thread Sequence Counting**: The Python-based sequence-to-FCGR counter takes $\sim 4.49\text{ s}$ per multi-megabase chromosome. A compiled C++/Rust or Cython k-mer counter would accelerate this upstream stage to sub-second speeds.
3. **Hardware Specificity**: Timings were obtained on an Intel Core i7-13650HX CPU on Windows 11. Multi-threaded GPU deployment or different CPU architectures will show different absolute throughputs, though relative dimensionality and memory ratios remain invariant.

---

## 8. Exact Output Paths & Reproduction Commands

### Primary Artifacts:
- Computational Benchmark Script: [`scripts/run_computational_cost_benchmark.py`](file:///d:/Image%20Research%20paper/scripts/run_computational_cost_benchmark.py)
- Figure Generation Script: [`scripts/generate_publication_figures.py`](file:///d:/Image%20Research%20paper/scripts/generate_publication_figures.py)
- Environment Metadata: [`runs/final_research_package/computational_cost/environment_metadata.json`](file:///d:/Image%20Research%20paper/runs/final_research_package/computational_cost/environment_metadata.json)
- Full Cost Results JSON: [`runs/final_research_package/computational_cost/computational_cost_results.json`](file:///d:/Image%20Research%20paper/runs/final_research_package/computational_cost/computational_cost_results.json)
- Cost Summary CSV: [`runs/final_research_package/computational_cost/computational_cost_summary.csv`](file:///d:/Image%20Research%20paper/runs/final_research_package/computational_cost/computational_cost_summary.csv)
- Cost Report Markdown: [`runs/final_research_package/computational_cost/computational_cost_report.md`](file:///d:/Image%20Research%20paper/runs/final_research_package/computational_cost/computational_cost_report.md)
- Figure Inventory: [`runs/final_research_package/figure_inventory.md`](file:///d:/Image%20Research%20paper/runs/final_research_package/figure_inventory.md)
- Figure Change Log: [`runs/final_research_package/figure_change_log.md`](file:///d:/Image%20Research%20paper/runs/final_research_package/figure_change_log.md)
- Revised Manuscript: [`runs/final_research_package/adaptive_cgr_vision_research_draft.md`](file:///d:/Image%20Research%20paper/runs/final_research_package/adaptive_cgr_vision_research_draft.md)
- Figure Directory (9 PNG + 9 Vector PDF): [`runs/final_research_package/figures/`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures)

### Reproduction Commands:
```powershell
# 1. Run the computational cost benchmark
python scripts/run_computational_cost_benchmark.py

# 2. Generate the complete publication figure suite (PNG + PDF)
python scripts/generate_publication_figures.py
```
