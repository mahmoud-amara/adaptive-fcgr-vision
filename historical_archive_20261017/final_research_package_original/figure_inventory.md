# Figure Inventory — Adaptive CGR-Vision Research Package

**Date**: 2026-10-12  
**Directory**: [`runs/final_research_package/figures/`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures)  
**Generation Script**: [`scripts/generate_publication_figures.py`](file:///d:/Image%20Research%20paper/scripts/generate_publication_figures.py)  
**Backup Directory**: [`runs/final_research_package/figures/backup_20261012_164106/`](file:///d:/Image%20Research%20paper/runs/final_research_package/figures/backup_20261012_164106)

---

## 1. Complete Figure Roster

| Manuscript Label | Final Filename (PNG + PDF) | Category | Description | Plotted Data Source | Status |
|---|---|---|---|---|---|
| **Figure 1** | `fig1_model_performance_bar` | Redesigned Benchmark | Horizontal bar chart of genus classification Macro-F1 (Mean ± SD) across representations with exact percentages. | `runs/final_research_package/final_benchmark_results.json` | ✅ Redesigned & Verified (PNG + PDF) |
| **Figure 2** | `fig2_ablation_family_f1` | Redesigned Ablation | Grouped bar chart comparing single feature families vs. combined 59D vector across both SVM-RBF and Random Forest. | `runs/ablation_study/ablation_results.json` | ✅ Redesigned & Verified (PNG + PDF) |
| **Figure 3** | `fig3_confusion_matrices` | Redesigned Matrix | Pooled out-of-fold confusion matrix ($N=150$) for primary model (*Multiscale 59D + RF*) with integer counts and recall percentages. | `runs/final_research_package/final_benchmark_results.json` | ✅ Redesigned & Verified (PNG + PDF) |
| **Figure 4** | `fig4_family_importance_distribution` | Redesigned Interpretability | Ranked horizontal contribution chart of ANOVA F-score family weights with dimensions and dominance indicators. | `runs/final_research_package/final_benchmark_results.json` | ✅ Redesigned & Verified (PNG + PDF) |
| **Figure 5 (Fig A)** | `fig_a_performance_vs_cost` | New Computational Cost | Macro-F1 vs. single-genome inference latency (log scale) Pareto scatter plot with representation sizes and optimal region callout. | `runs/final_research_package/computational_cost/computational_cost_results.json` | ✅ Created (PNG + PDF) |
| **Figure 6 (Fig B)** | `fig_b_stage_by_stage_runtime` | New Computational Cost | Dual-panel runtime chart: Sequence-to-FCGR generation (seconds scale) and downstream feature extraction + inference (ms scale). | `runs/final_research_package/computational_cost/computational_cost_results.json` | ✅ Created (PNG + PDF) |
| **Figure 7 (Fig C)** | `fig_c_memory_and_storage_footprint` | New Computational Cost | Dual-panel memory analysis: Feature matrix disk/RAM footprint (KB, $69.4\times$ reduction) and peak training process memory. | `runs/final_research_package/computational_cost/computational_cost_results.json` | ✅ Created (PNG + PDF) |
| **Figure 8 (Fig D)** | `fig_d_performance_cost_tradeoff` | New Computational Cost | 4-panel multi-criteria trade-off dashboard: Quality (F1), Storage (KB), Training time (ms), and Batch-30 Throughput (genomes/s). | `runs/final_research_package/computational_cost/computational_cost_results.json` | ✅ Created (PNG + PDF) |
| **Figure 9 (Fig E)** | `fig_e_pipeline_schematic` | New Architecture | Full end-to-end scientific architecture schematic: FastA $\to$ CGR $\to$ FCGR $\to$ 4 Feature Families $\to$ 59D $\to$ StandardScaler $\to$ Classifier $\to$ Genus. | Pipeline implementation (`src/adaptive_cgr_vision/`) | ✅ Created (PNG + PDF) |

---

## 2. Plotted Data Value Verification Audit

Every plotted point, bar, and annotation in the figures was audited against the stored artifacts:

1. **Figure 1**:
   - `Raw_FCGR_4096D__SVM_RBF`: Plotted $0.9472 \pm 0.0334$ (Acc: $94.7\%$). Verified in `final_benchmark_results.json` (`mean_macro_f1`: 0.947205, `std_macro_f1`: 0.033365).
   - `Raw_FCGR_4096D__RandomForest`: Plotted $0.9332 \pm 0.0204$ (Acc: $93.3\%$). Verified (`mean_macro_f1`: 0.933232, `std_macro_f1`: 0.020402).
   - `GLCM_18D__RandomForest`: Plotted $0.9331 \pm 0.0371$ (Acc: $93.3\%$). Verified (`mean_macro_f1`: 0.933131, `std_macro_f1`: 0.037083).
   - `Multiscale_59D__RandomForest`: Plotted $0.9246 \pm 0.0267$ (Acc: $92.7\%$). Verified (`mean_macro_f1`: 0.924646, `std_macro_f1`: 0.026681).
   - `Multiscale_59D__SVM_RBF`: Plotted $0.9195 \pm 0.0340$ (Acc: $92.0\%$). Verified (`mean_macro_f1`: 0.919461, `std_macro_f1`: 0.033981).
   - `Frequency_18D__RandomForest`: Plotted $0.9197 \pm 0.0267$ (Acc: $92.0\%$). Verified (`mean_macro_f1`: 0.919747, `std_macro_f1`: 0.026674).
   - `Spatial_12D__RandomForest`: Plotted $0.9103 \pm 0.0365$ (Acc: $91.3\%$). Verified (`mean_macro_f1`: 0.910320, `std_macro_f1`: 0.036545).
   - `Mash_1NN__Distance_Baseline`: Plotted $0.8770 \pm 0.0373$ (Acc: $87.3\%$). Verified (`mean_macro_f1`: 0.876958, `std_macro_f1`: 0.037274).
   - `Edge_11D__RandomForest`: Plotted $0.8561 \pm 0.0652$ (Acc: $86.0\%$). Verified (`mean_macro_f1`: 0.856086, `std_macro_f1`: 0.065153).
   - `Spatial_12D__SVM_RBF`: Plotted $0.8509 \pm 0.0337$ (Acc: $85.3\%$). Verified (`mean_macro_f1`: 0.850875, `std_macro_f1`: 0.033689).

2. **Figure 2 (Ablation)**:
   - Matches `runs/ablation_study/ablation_results.json` exactly:
     * Spatial 12D: SVM F1 = 0.8509 ± 0.0337 | RF F1 = 0.9103 ± 0.0365
     * Edge 11D: SVM F1 = 0.7851 ± 0.0424 | RF F1 = 0.8561 ± 0.0652
     * GLCM 18D: SVM F1 = 0.7807 ± 0.0643 | RF F1 = 0.9331 ± 0.0371
     * Frequency 18D: SVM F1 = 0.7343 ± 0.1024 | RF F1 = 0.9197 ± 0.0267
     * Combined 59D: SVM F1 = 0.9195 ± 0.0340 | RF F1 = 0.9246 ± 0.0267

3. **Figure 3 (Confusion Matrix)**:
   - Matches pooled out-of-fold prediction array in `final_benchmark_results.json`:
     * *Bacillus* ($N=25$): 19 correct (76% recall), 2 Esc, 1 Pse, 1 Sal, 2 Sta.
     * *Escherichia* ($N=25$): 25 correct (100% recall), 0 errors.
     * *Mycobacterium* ($N=25$): 22 correct (88% recall), 1 Pse, 2 Sal.
     * *Pseudomonas* ($N=25$): 25 correct (100% recall), 0 errors.
     * *Salmonella* ($N=25$): 24 correct (96% recall), 1 Bac.
     * *Staphylococcus* ($N=25$): 24 correct (96% recall), 1 Bac.

4. **Figure 4 (ANOVA Weights)**:
   - Matches `final_validation_audit.md` and feature weights: Spatial = 53.2%, GLCM = 15.6%, Edge = 13.1%, DCT = 9.2%, Wavelet = 8.9%.

5. **Figures A, B, C, D (Cost & Efficiency)**:
   - Matches `runs/final_research_package/computational_cost/computational_cost_results.json` exactly:
     * Raw FCGR 4096D Storage = 4,800.0 KB (float64)
     * Multiscale 59D Storage = 69.1 KB (float64, 69.42× reduction)
     * Multiscale 59D Feature extraction time = 8.05 ms/genome (124.2 genomes/sec)
     * Single inference latency (median): Multiscale SVM = 0.129 ms vs Raw SVM = 0.327 ms (2.5× speedup)
     * Training peak memory: Multiscale SVM = 63.2 KB vs Raw SVM = 3,850.2 KB (60.9× reduction)
     * Training peak memory: Multiscale RF = 205.9 KB vs Raw RF = 2,165.7 KB (10.5× reduction)
     * Batch-30 throughput: Multiscale SVM = 80,578 genomes/s vs Raw SVM = 5,640 genomes/s (14.3× speedup)
