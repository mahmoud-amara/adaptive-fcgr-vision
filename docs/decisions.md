# Adaptive CGR-Vision — Decisions Log

All research decisions that affect scientific validity are recorded here with date,
question, options considered, decision, rationale, approver, and affected config version.

The coding agent may recommend options and implement reversible scaffolding, but may NOT
silently change: dataset/task labels, inclusion rules, grouping/split, primary endpoint,
hypothesis/effect threshold, feature definitions, primary fusion method, confirmatory
search space, or interpretation of findings.

---

## Approved Decisions

### PD-01: Target taxonomic level
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** Species or genus classification?
- **Decision:** **Genus-level classification across 6 bacterial genera.**
- **Rationale:** Balanced class sizes, clinically/ecologically distinct clades with wide GC dynamic range (32.8% to 66.5%) and challenging sister taxa (*Escherichia* vs *Salmonella*).
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `dataset.target_rank = "genus"`

### PD-02: Organism panel
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** Which bacterial taxa to include? How many classes?
- **Decision:** **6 Genera ($N=150$ total genomes, 25 per genus):**
  1. *Escherichia* (25 complete chromosomes)
  2. *Salmonella* (25 complete chromosomes)
  3. *Pseudomonas* (25 complete chromosomes)
  4. *Bacillus* (25 complete chromosomes)
  5. *Staphylococcus* (25 complete chromosomes)
  6. *Mycobacterium* (25 complete chromosomes)
- **Rationale:** Provides high taxonomic diversity (Firmicutes, Actinobacteria, Proteobacteria), broad GC range, and fine-grained discrimination between Enterobacteriaceae sisters.
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `dataset.organism_panel`

### PD-03: Dataset source and release
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** Which NCBI RefSeq release or public benchmark?
- **Decision:** **NCBI RefSeq Complete Reference/Representative Genomes** with SHA-256 sequence verification.
- **Rationale:** High assembly accuracy, standardized curation, public domain license.
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `dataset.source = "ncbi_refseq"`

### PD-04: Assembly-level inclusion
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** Which assembly levels to include?
- **Decision:** **`Complete Genome` and `Chromosome` level only.**
- **Rationale:** Avoids fragmented scaffold/contig artificial boundary artifacts.
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `dataset.assembly_levels`

### PD-05: Multi-replicon handling
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** How to handle plasmids and multiple contigs/chromosomes per assembly?
- **Decision:** **Primary chromosome only (plasmids excluded).**
- **Rationale:** Mobile plasmid elements vary independently of lineage and introduce noise/confounding.
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `dataset.replicon_policy = "primary_chromosome_only"`

### PD-06: Quality thresholds
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** Maximum ambiguity fraction, minimum sequence length for inclusion?
- **Decision:** **Minimum Length = $1.0\text{ Mbp}$, Maximum Ambiguity Fraction = $0.01$ ($1.0\%$).**
- **Rationale:** Eliminates low-quality or severely truncated assemblies.
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `dataset.max_ambiguity_fraction = 0.01`, `dataset.min_length_bp = 1000000`

### PD-07: Similarity clustering tool and parameters
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** Which tool and identity/coverage cutoff for grouping?
- **Decision:** **MinHash / Mash distance single-linkage clustering at $\ge 95\%$ ANI.**
- **Rationale:** Guarantees 0% near-identical strain or homology leakage across cross-validation folds.
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `splitting.grouping.method = "minhash_ani"`, `threshold = 0.95`

### PD-08: Primary FCGR order (k)
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** Which k value for the primary FCGR representation?
- **Decision:** **$k=6$ ($64 \times 64$ grid, 4,096 cells).**
- **Rationale:** Optimal tradeoff between rich spatial texture resolution and k-mer count density for bacterial genomes.
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `representation.k = 6`

### PD-09: GLCM quantization parameters
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** Number of gray levels, binning strategy for GLCM?
- **Decision:** **32 gray levels with $\log(1+x)$ transform, distances $d \in \{1, 2, 3\}$, 4 angles aggregated.**
- **Rationale:** Balances co-occurrence matrix density and sensitivity to local textural gradients.
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `features.families.glcm.quantization_levels = 32`

### PD-10: Adaptive fusion formulation
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** Weight search space bounds and fusion strategy?
- **Decision:** **Inner 4-fold CV simplex grid search over family weight vector $\mathbf{w} \in \Delta^3$.**
- **Rationale:** Truly optimizes classification loss fold-locally without any validation or test fold lookahead.
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `fusion.primary_method = "simplex_grid_search"`

### PD-11: Smith–Waterman implementation
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** Which SW implementation, scoring parameters?
- **Decision:** **Dual Alignment Baseline: (1) Exact SW 1-NN on 16S rRNA genes (`Bio.Align.PairwiseAligner`), and (2) Whole-genome FastANI/BLAST+ 1-NN.**
- **Rationale:** Resolves the computational intractability of full-genome $O(N^2)$ SW while providing exact local alignment on canonical markers.
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `models.alignment_baselines`

### PD-12: Test allocation and CV folds
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** Held-out test %, outer/inner CV folds?
- **Decision:** **5-fold outer Grouped CV (80% train, 20% test per fold) with nested 4-fold inner Grouped CV on training folds.**
- **Rationale:** Maximizes sample utilization while strictly isolating model selection and hyperparameter tuning.
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `splitting.outer_cv_folds = 5`, `splitting.inner_cv_folds = 4`

### PD-13: Practical effect threshold
- **Date raised:** 2026-10-09
- **Date approved:** 2026-10-09
- **Question:** Minimum meaningful macro-F1 improvement?
- **Decision:** **$\Delta \text{Macro-F1} \ge 0.03$ (3.0% absolute improvement) with paired permutation test $p < 0.05$.**
- **Rationale:** Pre-specifies a clinically and practically meaningful superiority threshold.
- **Approver:** USER
- **Status:** ✅ APPROVED
- **Affected Config:** `configs/confirmatory.yaml` -> `evaluation.practical_effect_threshold = 0.03`

---

## Pending Decisions

*(None. All PD-01 through PD-13 approved by USER on 2026-10-09.)*
