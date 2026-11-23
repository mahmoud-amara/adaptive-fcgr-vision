# Computational Cost & Efficiency Analysis Report
**Adaptive CGR-Vision Research Package**  
**Date**: 2026-10-12  
**Hardware Environment**: 13th Gen Intel(R) Core(TM) i7-13650HX (14 cores / 20 logical threads), 16.0 GB RAM  
**Operating System**: Windows-11-10.0.26300-SP0  
**Software**: Python 3.14.3, NumPy 2.4.4, Scikit-Learn 1.9.1, SciPy 1.18.1

---

## 1. Executive Summary

This computational cost analysis provides empirical measurements of every stage in the Adaptive CGR-Vision pipeline, comparing the proposed **59-dimensional multiscale computer vision representation** against the baseline **raw 4,096-pixel FCGR grid** and traditional alignment-free distance sketching (**Mash 1-NN**).

### Key Empirical Findings:
1. **$69.4\times$ Feature Storage Reduction**: The 59D descriptor matrix for 150 genomes occupies **69.1 KB** (float64) compared to **4,915.2 KB (4.80 MB)** for the raw 4,096-pixel matrix ($69.42\times$ compression).
2. **Matched Classifier Latency Reduction**: For the matched kernel architecture (SVM-RBF), single-genome inference latency on 59D is **0.134 ms** (median **0.129 ms**) versus **0.334 ms** (median **0.327 ms**) on 4,096D—a **2.5× latency reduction** resulting from fewer kernel dot-product operations.
3. **Training Speedup**: Training SVM-RBF on 59D takes **5.0 ms** versus **17.9 ms** on raw FCGR, a **3.6× training speedup**.
4. **Transparent Full-Pipeline Accounting**: DNA sequence-to-FCGR grid generation requires **~3.41 seconds per complete bacterial chromosome** (~0.85 s/Mbp). Once FCGR grids are cached or streamed, 59D multiscale feature extraction executes in **8.06 ms per genome** (124.2 genomes/sec).

---

## 2. Representation Dimensionality & Storage Footprint

| Representation | Dimensions ($D$) | Numeric Dtype | Bytes / Genome | 150-Genome Matrix Size | Compression Ratio vs Raw |
|---|---|---|---|---|---|
| **Raw FCGR Grid** | 4,096 | `float64` | 32,768 B (32 KB) | **4,915.20 KB (4.80 MB)** | 1.0× (Baseline) |
| **Multiscale Vision (Ours)** | **59** | `float64` | **472 B (0.46 KB)** | **70.80 KB (0.07 MB)** | **69.42× Reduction** |
| **GLCM Texture Only** | 18 | `float64` | 144 B | 21.60 KB | 227.56× Reduction |
| **Spatial Moments Only** | 12 | `float64` | 96 B | 14.40 KB | 341.33× Reduction |
| **Mash MinHash Sketch** | 1,000 | `uint64` | 8,000 B | 1,200.00 KB (1.17 MB) | 4.10× Reduction |

---

## 3. Detailed Model Cost & Predictive Performance

All metrics measured across 5 outer cross-validation folds ($N_{train} = 120, N_{test} = 30$ per fold).

| Model Configuration | Feature Dim | Test Macro-F1 (Mean ± SD) | Test Accuracy (Mean ± SD) | Train Time (Mean ms) | Single Inference Latency (Median ms) | Batch-30 Throughput (Genomes/s) | Peak Heap Alloc. (KB) |
|---|---|---|---|---|---|---|---|
| **Raw FCGR + SVM-RBF** (Top 1) | 4,096 | 0.9472 ± 0.0334 | 94.67% | 17.94 | 0.327 | 5640.1 | 3850.2 |
| **Multiscale 59D + SVM-RBF** (Matched) | 59 | 0.9195 ± 0.0340 | 92.00% | 5.01 | 0.129 | 80578.2 | 63.2 |
| **Raw FCGR + RandomForest** (Top 2) | 4,096 | 0.9332 ± 0.0204 | 93.33% | 846.41 | 5.997 | 4617.0 | 2165.7 |
| **Multiscale 59D + RandomForest** (Proposed) | 59 | 0.9246 ± 0.0267 | 92.67% | 822.05 | 11.170 | 2268.1 | 205.9 |
| **GLCM 18D + RandomForest** | 18 | 0.9331 ± 0.0371 | 93.33% | 2563.42 | 26.992 | 1039.7 | 185.3 |
| **Mash 1-NN Baseline** | 1,000 | 0.8770 ± 0.0373 | 87.33% | 0.31 | 80.854 | 11.8 | 11.2 |

---

## 4. Stage-by-Stage Processing Time Breakdown

```
[Full Chromosome Sequence (1.3 - 6.5 Mbp)]
       │
       ▼  Stage 1: Sequence-to-FCGR Matrix Generation (~3.41 s / genome)
[64 x 64 FCGR Matrix (4,096 Pixels)]
       │
       ├──────────────────────────────────────────┬──────────────────────────────────────┐
       ▼ (Raw Branch)                             ▼ (Multiscale Branch)                  ▼ (Single Family)
[Flatten to 4,096D Vector]             [59D Feature Extraction]               [18D GLCM Extraction]
 (0.002 ms / genome)                    (8.06 ms / genome)             (3.14 ms / genome)
       │                                          │                                      │
       ▼ (StandardScaler Transform)               ▼ (StandardScaler Transform)           ▼ (StandardScaler Transform)
       │                                          │                                      │
       ▼ (Inference: SVM or RF)                   ▼ (Inference: SVM or RF)               ▼ (Inference: RF)
   SVM: 0.33 ms | RF: 6.00 ms     SVM: 0.13 ms | RF: 11.17 ms   RF: 26.99 ms
```

---

## 5. Reproduction Command
To reproduce this entire computational cost benchmark:
```powershell
python scripts/run_computational_cost_benchmark.py
```
