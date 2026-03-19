# PointNet Semantic Segmentation on Toronto-3D

**Exam project: Hands-on Python for Data Science**  
[2nd Level Master in Data Science and Statistical Learning (MD2SL)](https://www.imtlucca.it/master-ii-livello-data-science-and-statistical-learning-md2sl)  
[IMT School for Advanced Studies Lucca](https://www.imtlucca.it) & [Florence Center for Data Science, University of Florence](https://www.unifi.it)  
Instructor: Prof. Fabio Pinelli

## Overview

Per-point semantic segmentation of urban LiDAR point clouds using **PointNet** (Qi et al., 2017) implemented in PyTorch, trained and evaluated on the **Toronto-3D** benchmark dataset.

The project follows a two-phase approach:
1. **V1 (Baseline)**: vanilla PointNet with standard cross-entropy loss and StepLR scheduler.
2. **V2 (Improved)**: data augmentation, weighted cross-entropy, cosine annealing with warm restarts, larger spatial blocks, and mixed-precision training.

V2 achieves **+43% mIoU** and **+37% accuracy** over the baseline. Full results and analysis are available in [RESULTS.md](RESULTS.md).

## Dataset

[Toronto-3D](https://github.com/WeikaiTan/Toronto-3D) (Tan et al., 2020): approximately 78M LiDAR points from mobile mapping over a 1 km stretch of road in Toronto, manually annotated with 9 semantic classes: Road, Road Marking, Natural, Building, Utility Line, Pole, Car, Fence, and Unclassified.

Download the `.ply` files (L001, L002, L003, L004) and place them in `data/toronto3d/`.

| Split | Area | Points |
|-------|------|--------|
| Train | L001 | 10,695,757 |
| Train | L002 | 16,066,282 |
| Train | L004 | 10,536,643 |
| Test  | L003 | 41,021,528 |
| **Total** | | **78,320,210** |

## Project Structure

```
pointnet-toronto3d/
├── src/
│   ├── dataset.py         # Toronto3DDataset, v1 (10m blocks, stride 5m)
│   ├── dataset_v2.py      # Toronto3DDatasetV2, augmentation, 20m blocks
│   ├── model.py           # PointNet architecture (shared)
│   ├── train.py           # Training loop v1 (CE loss, StepLR)
│   ├── train_v2.py        # Training loop v2 (weighted CE, cosine annealing, AMP)
│   └── utils.py           # Visualization utilities
├── results/               # V1 outputs (plots, CSV, model weights)
├── results_v2/            # V2 outputs
├── run_explore.py         # Data exploration script
├── run_pipeline.py        # V1 full pipeline
├── run_pipeline_v2.py     # V2 full pipeline
├── job.slurm              # SLURM job script (v1)
├── job_v2.slurm           # SLURM job script (v2)
├── requirements.txt
├── RESULTS.md             # Detailed results and analysis
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

### SLURM (cluster)

Submit the job scripts to the SLURM scheduler. Each job runs exploration + full pipeline end-to-end, saving all outputs to `results/` or `results_v2/` and logs to `logs/`.

```bash
sbatch job.slurm       # v1 baseline (12h, 2× GPU)
sbatch job_v2.slurm    # v2 improved (24h, 2× GPU)
```

Monitor progress:
```bash
squeue -u "$USER"
tail -f logs/<JOBID>_output.log
```

### Interactive (local)

```bash
source .venv/bin/activate

# Data exploration: generates distribution plots, 3D visualizations, correlation matrix
python run_explore.py --data-dir data/toronto3d --results-dir results

# V1 baseline pipeline: grid search, cross-validation, final training, evaluation
python run_pipeline.py --data-dir data/toronto3d --results-dir results

# V2 improved pipeline: same stages with augmentation, weighted loss, cosine annealing
python run_pipeline_v2.py --data-dir data/toronto3d --results-dir results_v2
```

## Pipeline

Both v1 and v2 follow the same structure:

1. **Data Exploration**: class distribution, 3D visualizations, feature analysis, correlation matrix
2. **Data Preparation**: spatial blocking (v1: 10×10m, v2: 20×20m), normalization, PyTorch Dataset
3. **Grid Search**: hyperparameter search over learning rate, batch size, num_points (12 combinations)
4. **Cross-Validation**: 5-fold CV to estimate generalization performance
5. **Final Training**: full training set (L001 + L002 + L004)
6. **Test Evaluation**: accuracy, mIoU, per-class IoU, confusion matrix, classification report

## Results Summary

| Metric | V1 (Baseline) | V2 (Improved) | Δ |
|--------|---------------|---------------|---|
| Overall Accuracy | 39.4% | **54.1%** | +37% |
| Mean IoU | 20.4% | **29.1%** | +43% |
| CV mIoU | 0.281 ± 0.017 | **0.340 ± 0.020** | +21% |

See [RESULTS.md](RESULTS.md) for detailed per-class metrics, grid search tables, and analysis.

## V2 Improvements

| # | Improvement | Details | File |
|---|-------------|---------|------|
| 1 | **Data Augmentation** | Random Z-rotation (0 to 2π), Gaussian jitter (σ=0.01, clip ±0.05), random scaling (0.9 to 1.1) | `src/dataset_v2.py` |
| 2 | **Weighted Cross-Entropy** | Inverse square-root class frequency weighting to rebalance minority classes | `src/train_v2.py` |
| 3 | **Cosine Annealing + Early Stopping** | CosineAnnealingWarmRestarts (T₀=20, T_mult=2), early stopping with patience=15 | `src/train_v2.py` |
| 4 | **Larger Spatial Blocks** | 20×20m blocks (stride 10m) with 4096 to 8192 points for broader spatial context | `src/dataset_v2.py` |

Additional GPU optimizations: AMP (mixed precision), AdamW with weight decay 1e-4, cuDNN benchmark, `pin_memory` + `non_blocking` transfers.

## Hardware

- **GPU**: 2× NVIDIA L40S (46 GB VRAM each)
- **RAM**: 64 GB
- **Multi-GPU**: `nn.DataParallel`
- **Cluster**: SLURM, `gpu` partition

## References

- Qi, C.R., Su, H., Mo, K., & Guibas, L.J. (2017). [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593). *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 652, 660.
- Tan, W., Qin, N., Ma, L., Li, Y., Du, J., Cai, G., Yang, K., & Li, J. (2020). [Toronto-3D: A Large-scale Mobile LiDAR Dataset for Semantic Segmentation of Urban Roadways](https://arxiv.org/abs/2003.08284). *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 797, 806.
