#!/bin/bash
#SBATCH --job-name=pointnet-v2
#SBATCH --output=logs/%j_v2_output.log
#SBATCH --error=logs/%j_v2_error.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu

# ── Setup ────────────────────────────────────────────────────────
set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results_v2

echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "Working dir: $(pwd)"

source .venv/bin/activate

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:"
python -c "
import torch
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  [{i}] {p.name} — {p.total_memory / 1e9:.1f} GB')
"

# ── Pipeline v2 ─────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "PIPELINE v2 — PointNet migliorato"
echo "=========================================="
echo "Miglioramenti: augmentation, focal loss, cosine annealing, early stopping, AMP"
echo ""

python run_pipeline_v2.py \
    --data-dir data/toronto3d \
    --results-dir results_v2 \
    --gs-epochs 10 \
    --cv-epochs 100 \
    --final-epochs 100 \
    --k-folds 5 \
    --num-workers 8

echo ""
echo "=========================================="
echo "COMPLETATO v2 — $(date)"
echo "=========================================="
