# Results

Detailed results for the PointNet semantic segmentation experiments on Toronto-3D.  
Back to [README](README.md).

---

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [V1 — Baseline Results](#v1--baseline-results)
  - [Grid Search](#v1-grid-search)
  - [Cross-Validation](#v1-cross-validation)
  - [Test Evaluation](#v1-test-evaluation)
  - [Analysis](#v1-analysis)
- [V2 — Improved Results](#v2--improved-results)
  - [Grid Search](#v2-grid-search)
  - [Cross-Validation](#v2-cross-validation)
  - [Test Evaluation](#v2-test-evaluation)
  - [Analysis](#v2-analysis)
- [V1 vs V2 Comparison](#v1-vs-v2-comparison)

---

## Dataset Overview

**Toronto-3D** contains ~78M LiDAR points from mobile mapping, split into 4 areas along a 1 km road segment in Toronto. The class distribution is highly imbalanced (max/min ratio = 111.3×).

| Split | Area | Points |
|-------|------|--------|
| Train | L001 | 10,695,757 |
| Train | L002 | 16,066,282 |
| Train | L004 | 10,536,643 |
| Test  | L003 | 41,021,528 |
| **Total** | | **78,320,210** |

### Class Distribution

| Class | % of Total |
|-------|-----------|
| Road | 53.2% |
| Building | 24.4% |
| Natural | 14.1% |
| Unclassified | 4.5% |
| Car | 1.6% |
| Pole | 1.1% |
| Fence | 1.0% |
| Utility Line | 0.5% |
| Road Marking | 0.5% |

---

## V1 — Baseline Results

**Configuration**: PointNet vanilla, CrossEntropyLoss (unweighted), StepLR (step=10, γ=0.5), 10×10m blocks (stride 5m), 2048 points/block, 30 epochs.

### V1 Grid Search

12 combinations, 5 epochs each. Search space: `batch_size` ∈ {16, 32}, `learning_rate` ∈ {0.001, 0.0005, 0.0001}, `num_points` ∈ {2048, 4096}.

| batch_size | learning_rate | num_points | val_acc | val_mIoU | val_loss |
|-----------|--------------|-----------|---------|----------|----------|
| 16 | 0.001 | 2048 | 0.2893 | 0.1064 | 1.5438 |
| 16 | 0.001 | 4096 | 0.2980 | 0.1043 | 1.4653 |
| 16 | 0.0005 | 2048 | 0.2723 | 0.1087 | 1.5290 |
| 16 | 0.0005 | 4096 | 0.2976 | 0.1214 | 1.4025 |
| 16 | 0.0001 | 2048 | 0.2754 | 0.1199 | 1.4479 |
| 16 | 0.0001 | 4096 | 0.3141 | 0.1217 | 1.5334 |
| **32** | **0.001** | **2048** | **0.3259** | **0.1286** | **1.4500** |
| 32 | 0.001 | 4096 | 0.2604 | 0.1197 | 1.3624 |
| 32 | 0.0005 | 2048 | 0.2574 | 0.1183 | 1.2956 |
| 32 | 0.0005 | 4096 | 0.2424 | 0.1123 | 1.3633 |
| 32 | 0.0001 | 2048 | 0.3352 | 0.1232 | 1.4125 |
| 32 | 0.0001 | 4096 | 0.2632 | 0.1025 | 1.4101 |

**Best**: batch_size=32, lr=0.001, num_points=2048 (mIoU=0.1286)

### V1 Cross-Validation

5-fold, 30 epochs per fold with StepLR scheduler.

| Fold | Best mIoU |
|------|-----------|
| 0 | 0.2941 |
| 1 | 0.3060 |
| 2 | 0.2739 |
| 3 | 0.2739 |
| 4 | 0.2566 |
| **Mean** | **0.2809 ± 0.0173** |

### V1 Test Evaluation

Evaluated on L003 (41M points, 505 blocks).

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **39.4%** |
| **Mean IoU** | **20.4%** |

#### Per-class results (V1)

| Class | IoU | Precision | Recall | F1-score | Support |
|-------|-----|-----------|--------|----------|---------|
| Unclassified | 0.1516 | 0.1756 | 0.5263 | 0.2634 | 165,647 |
| Road | 0.1230 | 0.8975 | 0.1247 | 0.2190 | 416,328 |
| Road Marking | 0.0290 | 0.0292 | 0.8106 | 0.0563 | 8,102 |
| Natural | 0.4052 | 0.6604 | 0.5118 | 0.5766 | 609,863 |
| Building | 0.3576 | 0.7193 | 0.4156 | 0.5268 | 712,790 |
| Utility Line | 0.3491 | 0.3827 | 0.7989 | 0.5175 | 16,915 |
| Pole | 0.1558 | 0.1760 | 0.5765 | 0.2697 | 54,916 |
| Car | 0.1925 | 0.2132 | 0.6650 | 0.3229 | 64,033 |
| Fence | 0.0685 | 0.0929 | 0.2064 | 0.1280 | 19,886 |

### V1 Analysis

- **Natural** (0.41) and **Building** (0.36) achieve the highest IoU thanks to distinctive geometric and color features.
- **Road** (0.12 IoU): very high precision (0.90) but extremely low recall (0.12) — the model is conservative, classifying as Road only when highly confident, missing most road points.
- **Road Marking** (0.03) and **Fence** (0.07) suffer from severe class scarcity (0.5% and 1.0% of the dataset respectively).
- Results are consistent with the literature for vanilla PointNet on Toronto-3D. PointNet++ reports 42–59% mIoU with local feature aggregation (set abstraction layers) that vanilla PointNet lacks.

---

## V2 — Improved Results

**Configuration**: PointNet with weighted CrossEntropyLoss (inverse √freq), CosineAnnealingWarmRestarts (T₀=20, T_mult=2), data augmentation (rotation Z, jitter, scaling), AdamW (wd=1e-4), AMP, 20×20m blocks (stride 10m), 4096–8192 points/block, 100 epochs + early stopping (patience=15).

### V2 Grid Search

12 combinations, 10 epochs each. Search space: `batch_size` ∈ {32, 64}, `learning_rate` ∈ {0.001, 0.0005, 0.0001}, `num_points` ∈ {4096, 8192}.

| batch_size | learning_rate | num_points | val_acc | val_mIoU | val_loss |
|-----------|--------------|-----------|---------|----------|----------|
| 32 | 0.001 | 4096 | 0.4974 | 0.1726 | 1.4427 |
| 32 | 0.001 | 8192 | 0.5357 | 0.2151 | 1.3338 |
| 32 | 0.0005 | 4096 | 0.5038 | 0.1820 | 1.4030 |
| 32 | 0.0005 | 8192 | 0.5080 | 0.1988 | 1.3509 |
| 32 | 0.0001 | 4096 | 0.4896 | 0.1737 | 1.4426 |
| 32 | 0.0001 | 8192 | 0.4831 | 0.1624 | 1.4703 |
| 64 | 0.001 | 4096 | 0.5450 | 0.1958 | 1.3816 |
| 64 | 0.001 | 8192 | 0.5595 | 0.2178 | 1.4066 |
| **64** | **0.0005** | **4096** | **0.5524** | **0.2188** | **1.3702** |
| 64 | 0.0005 | 8192 | 0.4975 | 0.1956 | 1.3820 |
| 64 | 0.0001 | 4096 | 0.5062 | 0.1621 | 1.5023 |
| 64 | 0.0001 | 8192 | 0.5121 | 0.1715 | 1.5178 |

**Best**: batch_size=64, lr=0.0005, num_points=4096 (mIoU=0.2188)

### V2 Cross-Validation

5-fold, up to 100 epochs per fold with CosineAnnealingWarmRestarts and early stopping (patience=15).

| Metric | Value |
|--------|-------|
| **Mean mIoU** | **0.3399 ± 0.0204** |

### V2 Test Evaluation

Evaluated on L003 (41M points, 505 blocks).

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **54.1%** |
| **Mean IoU** | **29.1%** |

#### Per-class results (V2)

| Class | IoU | Precision | Recall | F1-score | Support |
|-------|-----|-----------|--------|----------|---------|
| Unclassified | 0.1617 | 0.1875 | 0.5403 | 0.2784 | 165,647 |
| Road | 0.5936 | 0.9482 | 0.6135 | 0.7450 | 416,328 |
| Road Marking | 0.1363 | 0.1484 | 0.6259 | 0.2400 | 8,102 |
| Natural | 0.4704 | 0.5931 | 0.6945 | 0.6398 | 609,863 |
| Building | 0.3460 | 0.6882 | 0.4104 | 0.5141 | 712,790 |
| Utility Line | 0.4219 | 0.4861 | 0.7617 | 0.5935 | 16,915 |
| Pole | 0.1877 | 0.3370 | 0.2975 | 0.3160 | 54,916 |
| Car | 0.2387 | 0.4646 | 0.3294 | 0.3855 | 64,033 |
| Fence | 0.0653 | 0.1046 | 0.1481 | 0.1227 | 19,886 |

### V2 Analysis

- **Road** shows the most dramatic improvement: IoU jumps from 0.12 to 0.59 (+383%). The larger 20m blocks provide much more spatial context, and the weighted loss rebalances training toward correctly classifying road surfaces.
- **Utility Line** (+21%), **Road Marking** (+370%), **Car** (+24%): augmentation and weighted loss significantly improve minority class performance.
- **Building** shows a slight decrease (0.36 → 0.35): a natural trade-off when rebalancing toward weaker classes.
- **Fence** remains the hardest class (0.07): only 1% of the dataset with geometrically ambiguous structures.
- The gap to PointNet++ (42–59% mIoU in literature) confirms that the architectural limitations of vanilla PointNet (no local feature hierarchy) remain the main bottleneck.

---

## V1 vs V2 Comparison

### Summary

| Metric | V1 (Baseline) | V2 (Improved) | Improvement |
|--------|---------------|---------------|-------------|
| GS best mIoU | 0.1286 | 0.2188 | +70% |
| CV mIoU | 0.2809 ± 0.017 | 0.3399 ± 0.020 | +21% |
| Overall Accuracy | 39.4% | **54.1%** | +37% |
| Mean IoU | 20.4% | **29.1%** | +43% |

### Per-class IoU Comparison

| Class | V1 | V2 | Δ |
|-------|-----|-----|---|
| Unclassified | 0.152 | 0.162 | +7% |
| Road | 0.123 | **0.594** | +383% |
| Road Marking | 0.029 | **0.136** | +370% |
| Natural | 0.405 | **0.470** | +16% |
| Building | 0.358 | 0.346 | −3% |
| Utility Line | 0.349 | **0.422** | +21% |
| Pole | 0.156 | **0.188** | +21% |
| Car | 0.193 | **0.239** | +24% |
| Fence | 0.069 | 0.065 | −5% |

### Key Takeaways

1. **Larger spatial blocks** (10m → 20m) are the single most impactful change, especially for spatially extended classes like Road.
2. **Weighted cross-entropy** effectively addresses the severe class imbalance (111× ratio), rescuing Road from 0.12 to 0.59 IoU.
3. **Data augmentation** improves robustness across all classes, particularly for minority classes with limited training samples.
4. **Cosine annealing + early stopping** allows longer training (100 vs 30 epochs) without overfitting, while reducing the need for manual LR schedule tuning.
5. The remaining gap to PointNet++ (~42–59% mIoU) reflects the fundamental architectural limitation of PointNet: no local neighborhood feature aggregation. Closing this gap would require hierarchical architectures (e.g., PointNet++, KPConv, RandLA-Net).
