# Results

Detailed results for the PointNet semantic segmentation experiments on Toronto-3D.  
Back to [README](README.md).

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [V1 (Baseline) Results](#v1--baseline-results)
  - [Grid Search](#v1-grid-search)
  - [Cross-Validation](#v1-cross-validation)
  - [Test Evaluation](#v1-test-evaluation)
  - [Analysis](#v1-analysis)
- [V2 (Improved) Results](#v2--improved-results)
  - [Grid Search](#v2-grid-search)
  - [Cross-Validation](#v2-cross-validation)
  - [Test Evaluation](#v2-test-evaluation)
  - [Analysis](#v2-analysis)
- [V1 vs V2 Comparison](#v1-vs-v2-comparison)

## Dataset Overview

The **Toronto-3D** dataset (Tan et al., CVPRW 2020) is a large-scale outdoor LiDAR point cloud benchmark for semantic segmentation. It was collected via a vehicle-mounted MLS (Mobile Laser Scanning) system along a 1 km stretch of Avenue Road in Toronto, Canada. The dataset contains approximately 78.3 million points, each annotated with one of 9 semantic classes.

Each point carries 7 features: 3D coordinates (x, y, z), return intensity, and RGB color values. The dataset is split into four contiguous areas (L001 to L004), with L003 held out as the official test set.

| Split | Area | Points |
|-------|------|--------|
| Train | L001 | 10,695,757 |
| Train | L002 | 16,066,282 |
| Train | L004 | 10,536,643 |
| Test  | L003 | 41,021,528 |
| **Total** | | **78,320,210** |

### Class Distribution

The class distribution is highly imbalanced: Road and Building together account for over 77% of all points, while Road Marking and Utility Line each represent only about 0.5%. The maximum-to-minimum class ratio is 111.3×. This imbalance is a central challenge for the segmentation task and directly motivates the use of weighted loss functions in V2.

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

## V1 (Baseline) Results

The V1 baseline uses a vanilla PointNet architecture with standard (unweighted) cross-entropy loss and a StepLR learning rate scheduler. The point cloud is divided into 10×10 m spatial blocks with 5 m stride, each subsampled to 2048 points. Training runs for 30 epochs.

**Configuration summary**: CrossEntropyLoss (unweighted), Adam optimizer, StepLR (step=10, γ=0.5), 10×10 m blocks (stride 5 m), 2048 points/block, 30 epochs, 2× NVIDIA L40S via DataParallel.

### V1 Grid Search

We performed a grid search over 12 hyperparameter combinations, training each for 5 epochs on a single train/validation split (L001+L002 for training, L004 for validation). The search space covered batch sizes of 16 and 32, learning rates of 0.001, 0.0005, and 0.0001, and point counts of 2048 and 4096 per block.

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

**Best combination**: batch_size=32, lr=0.001, num_points=2048 (val mIoU = 0.1286). The results show only modest variation across combinations, suggesting the model's capacity is the main bottleneck at this stage rather than hyperparameter tuning.

### V1 Cross-Validation

Using the best hyperparameters from grid search, we ran 5-fold cross-validation on the training areas (L001, L002, L004). Each fold trained for 30 epochs with StepLR. The purpose of cross-validation is to obtain a more robust estimate of generalization performance and to assess variance across different data splits.

| Fold | Best mIoU |
|------|-----------|
| 0 | 0.2941 |
| 1 | 0.3060 |
| 2 | 0.2739 |
| 3 | 0.2739 |
| 4 | 0.2566 |
| **Mean** | **0.2809 ± 0.0173** |

The relatively low standard deviation (0.017) indicates stable performance across folds, suggesting the model generalizes consistently despite the limited expressiveness of vanilla PointNet.

### V1 Test Evaluation

After cross-validation, we trained a final model on the entire training set (L001 + L002 + L004) using the best hyperparameters, and evaluated it on the held-out test area L003 (41 million points, 505 spatial blocks).

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

The V1 baseline results reveal several clear patterns that informed the design of V2:

- **Natural** (IoU 0.41) and **Building** (IoU 0.36) are the best-performing classes. Both have distinctive geometric profiles and color signatures that make them separable even without local feature aggregation.
- **Road** stands out as a paradox: very high precision (0.90) but extremely low recall (0.12), yielding IoU of just 0.12. The unweighted loss function causes the model to be dominated by the majority class (Road = 53% of all points) during training, but in a counterintuitive way: the model learns to be very conservative, classifying a point as Road only when it is highly confident. This results in missing the vast majority of road points. The small 10 m block size likely exacerbates this issue, as road surfaces lack distinctive local features in such small windows.
- **Road Marking** (IoU 0.03) and **Fence** (IoU 0.07) are the worst performers, suffering from extreme class scarcity (0.5% and 1.0% of the dataset). With only 8,102 Road Marking points in the test set, the model has insufficient examples to learn robust features.
- **Utility Line** (IoU 0.35) performs surprisingly well despite having only 0.5% of points, likely because overhead wires have a very distinctive spatial geometry (thin, elevated, spanning between poles).

Overall, the V1 results (20.4% mIoU) are consistent with what the literature reports for vanilla PointNet on Toronto-3D. More advanced architectures like PointNet++, which introduce hierarchical local feature aggregation through set abstraction layers, achieve 42 to 59% mIoU on this benchmark.

## V2 (Improved) Results

The V2 pipeline introduces four targeted improvements designed to address the specific weaknesses observed in V1: (1) data augmentation for better generalization, (2) inverse-square-root weighted cross-entropy loss to address class imbalance, (3) cosine annealing with warm restarts and early stopping for more effective learning rate scheduling, and (4) larger spatial blocks (20×20 m) to provide broader context.

Additional GPU-level optimizations include automatic mixed precision (AMP) for faster training, AdamW optimizer with weight decay for better regularization, cuDNN benchmark mode, and optimized CPU-to-GPU data transfers.

**Configuration summary**: weighted CrossEntropyLoss (inverse √freq), AdamW (wd=1e-4), CosineAnnealingWarmRestarts (T₀=20, T_mult=2), early stopping (patience=15), data augmentation (Z-rotation, jitter, scaling), AMP (FP16), 20×20 m blocks (stride 10 m), 4096 to 8192 points/block, up to 100 epochs, 2× NVIDIA L40S via DataParallel.

### V2 Grid Search

We expanded the grid search to use 10 epochs per combination (vs. 5 in V1) and adjusted the search space to reflect the larger block sizes: batch sizes of 32 and 64, learning rates of 0.001, 0.0005, and 0.0001, and point counts of 4096 and 8192.

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

**Best combination**: batch_size=64, lr=0.0005, num_points=4096 (val mIoU = 0.2188). The best GS mIoU already improved by +0.090 over V1 (0.1286), confirming that the V2 changes have a measurable impact even within the first 10 training epochs. Interestingly, 4096 points outperformed 8192 for the best learning rate, suggesting diminishing returns from denser sampling when the block size is already 20 m.

### V2 Cross-Validation

We ran 5-fold cross-validation with up to 100 epochs per fold, using CosineAnnealingWarmRestarts as the learning rate scheduler and early stopping with patience of 15 epochs. This allows the model to train significantly longer than V1 (30 epochs) while avoiding overfitting.

| Metric | Value |
|--------|-------|
| **Mean mIoU** | **0.3399 ± 0.0204** |

The V2 CV mIoU of 0.340 represents a +0.059 improvement over V1 (0.281). The standard deviation remains similar (0.020 vs 0.017), indicating that the improvements are consistent across folds.

### V2 Test Evaluation

The final V2 model was trained on the full training set and evaluated on L003.

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

The V2 improvements produce consistent gains across almost all classes, with particularly striking results on Road:

- **Road** (IoU +0.471, from 0.12 to 0.59) shows the largest absolute improvement. The combination of larger blocks (providing 4× the spatial context) and weighted loss (correcting the training imbalance) transforms Road from one of the worst classes to the best. In V1, recall was just 0.12; in V2 it rises to 0.61, while precision remains very high at 0.95.
- **Road Marking** (IoU +0.107, from 0.03 to 0.14) benefits from the weighted loss that upweights this extremely rare class, and from the augmentations that effectively multiply its training samples.
- **Utility Line** (IoU +0.073, from 0.35 to 0.42) and **Car** (IoU +0.046, from 0.19 to 0.24) also improve, as the rebalanced loss allocates more gradient signal to these minority classes.
- **Natural** (IoU +0.065, from 0.41 to 0.47) improves thanks to data augmentation, which helps the model generalize better to varied vegetation appearances.
- **Building** (IoU −0.012, from 0.36 to 0.35) and **Fence** (IoU −0.004, from 0.07 to 0.07) show marginal decreases. This is the expected trade-off: rebalancing the loss toward rare classes slightly reduces performance on classes that were already well-represented.
- **Fence** remains the most difficult class at 0.07 IoU: it has very few samples (1% of the dataset) and is geometrically similar to other vertical structures (poles, building edges).

The overall gap to PointNet++ (42 to 59% mIoU in literature) confirms that the main bottleneck is now architectural: vanilla PointNet processes each point independently through shared MLPs and relies solely on a global max-pooling operation, which cannot capture local geometric patterns at different scales.

## V1 vs V2 Comparison

### Summary

The table below summarizes the overall improvements from V1 to V2. We report absolute differences: percentage points (pp) for metrics expressed as percentages, and raw differences for mIoU values.

| Metric | V1 (Baseline) | V2 (Improved) | Δ |
|--------|---------------|---------------|---|
| GS best mIoU | 0.1286 | 0.2188 | +0.090 |
| CV mIoU | 0.2809 ± 0.017 | 0.3399 ± 0.020 | +0.059 |
| Overall Accuracy | 39.4% | **54.1%** | +14.7 pp |
| Mean IoU | 20.4% | **29.1%** | +8.7 pp |

### Per-class IoU Comparison

The following table compares per-class IoU between V1 and V2. The delta column shows absolute differences.

| Class | V1 | V2 | Δ (absolute) |
|-------|-----|-----|---|
| Unclassified | 0.152 | 0.162 | +0.010 |
| Road | 0.123 | **0.594** | +0.471 |
| Road Marking | 0.029 | **0.136** | +0.107 |
| Natural | 0.405 | **0.470** | +0.065 |
| Building | 0.358 | 0.346 | −0.012 |
| Utility Line | 0.349 | **0.422** | +0.073 |
| Pole | 0.156 | **0.188** | +0.032 |
| Car | 0.193 | **0.239** | +0.046 |
| Fence | 0.069 | 0.065 | −0.004 |

Seven out of nine classes improve, with Road (+0.471) and Road Marking (+0.107) showing the largest absolute gains. The two classes that slightly decrease (Building −0.012, Fence −0.004) show negligible losses, well within the expected trade-off of loss rebalancing.

### Key Takeaways

1. **Larger spatial blocks** (10 m to 20 m) are the single most impactful change, especially for spatially extended classes like Road. The 4× increase in block area provides the model with significantly more context to distinguish flat surfaces from one another.
2. **Weighted cross-entropy** effectively addresses the severe class imbalance (111× ratio). By upweighting rare classes with inverse square-root frequency weights, the loss function ensures that minority classes contribute meaningful gradients during training.
3. **Data augmentation** (random rotation, jitter, scaling) improves robustness across all classes, particularly for minority classes with limited training samples. It effectively multiplies the training set diversity without additional data collection.
4. **Cosine annealing with warm restarts + early stopping** enables longer training (up to 100 epochs vs. 30) without overfitting, while removing the need for manual learning rate schedule tuning.
5. The remaining gap to PointNet++ (approx. 42 to 59% mIoU in literature) reflects the fundamental architectural limitation of vanilla PointNet: it lacks local neighborhood feature aggregation. Closing this gap would require hierarchical architectures such as PointNet++, KPConv, or RandLA-Net that can capture geometric patterns at multiple spatial scales.
