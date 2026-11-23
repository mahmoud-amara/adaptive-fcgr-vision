# Figure Change Log — Adaptive CGR-Vision Research Package

**Date**: 2026-10-12  
**Purpose**: Document all aesthetic, scientific, and structural changes made to the figure suite, following the publication-grade figure overhaul instructions.

---

## 1. Summary of Changes

| Figure | Prior State | Action Taken | Rationale & Scientific Enhancements |
|---|---|---|---|
| **Figure 1** (`fig1_model_performance_bar`) | Basic PNG with generic dark colors, limited labels. | **Complete Redesign** (300 DPI PNG + Vector PDF) | • Applied Okabe-Ito colorblind-safe palette.<br>• Displayed both Macro-F1 (Mean ± SD) and test accuracy percentages.<br>• Added benchmark reference line for raw grid baseline.<br>• Exported vector PDF for crystal-clear publication scaling. |
| **Figure 2** (`fig2_ablation_family_f1`) | Only showed RandomForest bars, neglecting kernel classifier comparison. | **Complete Redesign** (300 DPI PNG + Vector PDF) | • Converted into dual-model grouped bar chart (SVM-RBF vs. RandomForest).<br>• Visually demonstrates the core finding: SVM requires multi-family synergy ($p_{\text{Holm}} \le 0.014$), while RF is more tolerant of single high-capacity families.<br>• Clean typography, error bars, value callouts. |
| **Figure 3** (`fig3_confusion_matrices`) | Basic seaborn heatmap with plain integer numbers and no recall indicators. | **Complete Redesign** (300 DPI PNG + Vector PDF) | • Italicized scientific genus labels (*Bacillus*, *Escherichia*, etc.).<br>• Added explicit per-class recall indicators `[XX%]` alongside row cells.<br>• Professional Blues colormap with high-contrast font choices.<br>• Exported vector PDF. |
| **Figure 4** (`fig4_family_importance_distribution`) | Basic matplotlib pie chart (discouraged in scientific journals due to angular distortion). | **Complete Redesign** (300 DPI PNG + Vector PDF) | • Replaced pie chart with a horizontal ranked contribution chart with dimensions.<br>• Clear percentage readouts and cumulative threshold line highlighting Spatial dominance (>50%).<br>• Exported vector PDF. |
| **Figure A** (`fig_a_performance_vs_cost`) | Did not exist. | **Newly Created** (300 DPI PNG + Vector PDF) | • Macro-F1 vs. single-genome inference latency (log scale) Pareto scatter plot.<br>• Marker shapes indicate classifier family (circle = SVM, square = RF, diamond = 1-NN).<br>• Point size encodes feature dimensions ($4096\text{D} \to 59\text{D}$).<br>• Highlights Pareto optimality of Multiscale 59D + SVM-RBF and Multiscale 59D + RandomForest. |
| **Figure B** (`fig_b_stage_by_stage_runtime`) | Did not exist. | **Newly Created** (300 DPI PNG + Vector PDF) | • Dual-panel chart separating seconds-scale sequence-to-FCGR generation ($3.15\text{--}5.86\text{ s}$) from milliseconds-scale downstream feature extraction + inference ($0.13\text{--}28\text{ ms}$).<br>• Accurately addresses timing accounting without confusing sequence parsing with matrix inference. |
| **Figure C** (`fig_c_memory_and_storage_footprint`) | Did not exist. | **Newly Created** (300 DPI PNG + Vector PDF) | • Dual-panel memory analysis comparing feature matrix storage ($4,800\text{ KB} \to 69.1\text{ KB}$, $69.4\times$ reduction) and peak training process RAM ($3,850\text{ KB} \to 63.2\text{ KB}$, $60.9\times$ reduction for SVM). |
| **Figure D** (`fig_d_performance_cost_tradeoff`) | Did not exist. | **Newly Created** (300 DPI PNG + Vector PDF) | • 4-panel comprehensive trade-off dashboard: Quality (F1), Storage (KB), Training time (ms), and Batch-30 Throughput (genomes/sec).<br>• Direct matched comparison between Raw 4096D and Multiscale 59D across both SVM and RF. |
| **Figure E** (`fig_e_pipeline_schematic`) | Did not exist. | **Newly Created** (300 DPI PNG + Vector PDF) | • Publication-grade vector schematic showing the complete pipeline: DNA FastA $\to$ $64 \times 64$ FCGR $\to$ 4 visual families $\to$ 59D Multiscale Vector $\to$ StandardScaler $\to$ Classifier $\to$ Taxonomic Genus. |

---

## 2. Publication Figure Captions & Descriptions

### Figure 1: Genus Taxonomic Classification Performance Across Representations
- **Caption**: *Figure 1. Comparison of taxonomic genus classification performance across representations and models on 150 complete RefSeq bacterial chromosomes under 5-fold grouped cross-validation (zero 95% ANI inter-fold leakage). Bars denote unweighted mean Macro-F1 score across the 5 outer test folds with standard deviation error bars. Values in parentheses denote overall classification accuracy. The dotted vertical line marks the baseline performance of raw 4,096-pixel FCGR grids with SVM-RBF.*
- **Demonstrates**: Multiscale 59D descriptors preserve classification accuracy ($92.67\%$ for RF, $92.00\%$ for SVM-RBF) relative to raw 4,096-pixel grids ($93.33\%$ for RF, $94.67\%$ for SVM-RBF) while achieving a $69.4\times$ dimensional reduction. Both outperform Mash 1-NN sketch distance ($87.33\%$).
- **Limitations**: Evaluated on 6 bacterial genera ($N=150$ genomes); does not establish cross-family generalization beyond the 6 curated classes.

### Figure 2: Feature Family Ablation Analysis (SVM-RBF vs. RandomForest)
- **Caption**: *Figure 2. Feature family ablation analysis comparing single-family vision descriptors against the combined 59-dimensional multiscale representation across kernel (SVM-RBF) and tree-ensemble (RandomForest) classifiers. Error bars denote standard deviation across 5 outer folds. For SVM-RBF, combining all four feature families yields statistically significant performance gains over any single family ($p_{\text{Holm}} \le 0.0140$).*
- **Demonstrates**: Complementary synergy among visual feature families is critical for kernel machines, where Macro-F1 increases from $0.734\text{--}0.851$ (single families) to $0.9195$ (combined 59D). Random Forest exhibits higher individual family capacity (GLCM alone achieves $0.9331$).
- **Limitations**: Feature families were partitioned at fixed extraction parameters; does not search all possible subset interactions.

### Figure 3: Out-of-Fold Taxonomic Confusion Matrix
- **Caption**: *Figure 3. Pooled out-of-fold confusion matrix for the primary Multiscale 59D + RandomForest model across all 150 RefSeq bacterial genomes ($N=25$ per genus). Numbers indicate exact genome counts. Values on the right margin indicate per-class sensitivity (recall).*
- **Demonstrates**: Perfect separation for *Escherichia* ($100\%$) and *Pseudomonas* ($100\%$), high recall for *Salmonella* ($96\%$) and *Staphylococcus* ($96\%$), and moderate confusion in *Bacillus* ($76\%$, 19/25), reflecting its broad morphological and phylogenetic diversity.
- **Limitations**: Reflects closed-set 6-genus classification only; does not evaluate open-set out-of-distribution rejection.

### Figure 4: Feature Family Importance Distribution (ANOVA F-Score Weighting)
- **Caption**: *Figure 4. Relative discriminative contribution of the five computer vision feature sub-families determined by fold-local ANOVA F-score weighting. Bar lengths reflect percentage of total discriminative score; numbers in parentheses denote feature count.*
- **Demonstrates**: Spatial Distribution moments contribute over half of the total discriminative signal ($53.2\%$), demonstrating that global spatial density and quadrant mass distribution drive genomic separation, followed by GLCM texture ($15.6\%$) and edge contrast ($13.1\%$).
- **Limitations**: Feature weights are empirical predictive associations derived from ANOVA F-statistic in this specific dataset and do not represent direct biological or causal phylogenetic weights.

### Figure A: Classification Quality vs. Inference Latency (Pareto Frontier)
- **Caption**: *Figure A. Pareto trade-off between predictive quality (Macro-F1) and single-genome inference latency (milliseconds, log scale). Point size encodes feature dimensionality. Marker shapes denote classifier families (circles: SVM-RBF; squares: RandomForest; diamond: Mash 1-NN). The shaded top-left region denotes the optimal operational regime.*
- **Demonstrates**: Multiscale 59D + SVM-RBF achieves the fastest single-genome latency ($0.129\text{ ms}$, $2.5\times$ faster than raw FCGR SVM) with strong classification accuracy ($0.9195$ F1). Multiscale 59D + RandomForest achieves high accuracy ($0.9246$ F1) at $11.17\text{ ms}$ latency. Mash 1-NN is $626\times$ slower in latency ($80.85\text{ ms}$) due to iterative sketch distance lookups.
- **Limitations**: Single-genome inference measured in Python process memory on single-thread CPU; latency under multi-threaded GPU serving may differ.

### Figure B: Computational Runtime Breakdown Across Pipeline Stages
- **Caption**: *Figure B. Stage-by-stage computational runtime analysis. Panel A illustrates upstream sequence-to-FCGR matrix construction time (seconds scale) across representative bacterial chromosomes of varying lengths ($1.3\text{--}6.5\text{ Mbp}$). Panel B illustrates downstream per-genome runtime (milliseconds scale) partitioned into feature extraction from precomputed FCGR grids and classifier inference latency.*
- **Demonstrates**: Sequence-to-FCGR matrix generation takes $\sim 4.49\text{ s}$ per full genome ($\sim 0.99\text{ s/Mbp}$). Downstream 59D feature extraction takes only $8.05\text{ ms}$ per genome, and SVM inference takes $0.13\text{ ms}$, enabling high-throughput classification once FCGR grids are generated or streamed.
- **Limitations**: Upstream parsing depends on disk I/O and sequence length; downstream timings assume in-memory floating-point grids.

### Figure C: Memory Footprint and Storage Compression Analysis
- **Caption**: *Figure C. Comparative memory requirements between raw and compact representations. Panel A compares total feature matrix storage in memory/disk for 150 genomes on a logarithmic scale ($69.4\times$ reduction from 4,800 KB to 69.1 KB). Panel B compares peak dynamic process memory allocated during model training ($60.9\times$ reduction for SVM-RBF).*
- **Demonstrates**: Compressing the representation from 4,096 to 59 features yields a massive reduction in matrix storage ($4.80\text{ MB} \to 0.07\text{ MB}$) and drastically reduces peak RAM during kernel machine training ($3,850\text{ KB} \to 63.2\text{ KB}$).
- **Limitations**: Peak RAM measured via Python `tracemalloc` heap tracking; underlying C/C++ library buffers (BLAS/LAPACK) may have small additional fixed footprints.

### Figure D: Multi-Criteria Trade-off: Raw FCGR vs. Multiscale 59D Descriptors
- **Caption**: *Figure D. Multi-criteria performance and cost comparison directly contrasting Raw 4,096-pixel FCGR grids against the proposed Multiscale 59D vision descriptors under matched classifier architectures (SVM-RBF and RandomForest). Panels display Macro-F1 (Panel A), Feature Matrix Storage (Panel B), Fold Training Time (Panel C), and Batch-30 Inference Throughput (Panel D).*
- **Demonstrates**: Multiscale 59D descriptors deliver a $69.4\times$ reduction in storage, a $3.6\times$ training speedup (SVM), and a $14.3\times$ boost in batch throughput ($80,578\text{ genomes/s}$ vs. $5,640\text{ genomes/s}$), with negligible loss in Macro-F1 ($\le 0.027$).
- **Limitations**: Batch throughput measured on in-memory feature matrices with batch size 30; real-world throughput in complete pipelines is bounded by upstream sequence I/O.

### Figure E: Adaptive CGR-Vision End-to-End Processing Architecture
- **Caption**: *Figure E. Scientific schematic of the Adaptive CGR-Vision architecture. Raw bacterial chromosome sequences are mapped into a $64 \times 64$ count-normalized FCGR grid ($k=6$), followed by deterministic extraction of 59 computer vision descriptors across four families: GLCM texture (18D), spatial distribution moments (12D), frequency subband energies (18D), and edge/gradient statistics (11D). The concatenated 59D vector is standardized strictly on training folds and passed to downstream classifiers for genus-level taxonomic prediction.*
- **Demonstrates**: End-to-end dataflow, dimensionality transformation ($4,096 \to 59$), strict zero-leakage training-fold scaling, and target taxonomic outputs.
- **Limitations**: Schematic diagram representing system design and mathematical mappings, not an empirical data plot.
