#!/bin/bash
#SBATCH --job-name=pointnet-toronto3d
#SBATCH --output=logs/%j_output.log
#SBATCH --error=logs/%j_error.log
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu

# ── Setup ────────────────────────────────────────────────────────
set -e
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results

echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "Working dir: $(pwd)"

# Attiva il virtual environment
source .venv/bin/activate

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA disponibile: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU disponibili: $(python -c 'import torch; print(torch.cuda.device_count())')"

# ── Step 1: Data Exploration ─────────────────────────────────────
echo ""
echo "=========================================="
echo "STEP 1: DATA EXPLORATION"
echo "=========================================="
python run_explore.py --data-dir data/toronto3d --results-dir results

# ── Step 2: ML Pipeline ─────────────────────────────────────────
echo ""
echo "=========================================="
echo "STEP 2: ML PIPELINE"
echo "=========================================="
python run_pipeline.py \
    --data-dir data/toronto3d \
    --results-dir results \
    --gs-epochs 5 \
    --cv-epochs 30 \
    --final-epochs 30 \
    --k-folds 5 \
    --num-workers 8

echo ""
echo "=========================================="
echo "COMPLETATO — $(date)"
echo "=========================================="
