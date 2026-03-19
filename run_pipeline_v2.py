#!/usr/bin/env python3
"""Pipeline v2 — Improved PointNet on Toronto-3D.

Optimized version with:
  1. Data augmentation (rotazione Z, jitter, scaling)
  2. Focal Loss (γ=2) for class imbalance  →  replaced with weighted CrossEntropyLoss
  3. CosineAnnealingWarmRestarts + Early Stopping
  4. Blocchi 20×20 m con 4096+ punti
  5. Mixed Precision (AMP) per 2× throughput
  6. AdamW con weight decay
"""

import os
import argparse
import sys
import traceback
import json
import numpy as np
import pandas as pd
import torch

torch.backends.cudnn.benchmark = True  # optimize convolutions for fixed input size

from torch.utils.data import DataLoader

from src.dataset import load_toronto3d_ply, NUM_CLASSES
from src.dataset_v2 import Toronto3DDatasetV2
from src.model import PointNetSegmentation
from src.train_v2 import (
    get_device,
    grid_search,
    cross_validate,
    train_final_model,
    evaluate,
    compute_class_weights,
    FocalLoss,
)
from src.utils import (
    plot_confusion_matrix,
    plot_iou_per_class,
    plot_cv_learning_curves,
    plot_prediction_comparison,
    print_test_results,
)


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline v2 — Improved PointNet on Toronto-3D',
    )
    parser.add_argument('--data-dir', default='data/toronto3d')
    parser.add_argument('--results-dir', default='results_v2')
    parser.add_argument('--gs-epochs', type=int, default=10,
                        help='Epochs for GridSearch')
    parser.add_argument('--cv-epochs', type=int, default=100,
                        help='Max epochs for Cross-Validation (with early stopping)')
    parser.add_argument('--final-epochs', type=int, default=100,
                        help='Epochs for final training')
    parser.add_argument('--k-folds', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} "
                  f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
    print(f"AMP (Mixed Precision): enabled")
    print(f"cuDNN benchmark: enabled")

    # ── Data loading ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATA LOADING")
    print("=" * 60)

    areas = {}
    for area_name in ['L001', 'L002', 'L003', 'L004']:
        filepath = os.path.join(args.data_dir, f'{area_name}.ply')
        if os.path.exists(filepath):
            print(f"  Loading {area_name}...")
            areas[area_name] = load_toronto3d_ply(filepath)
            print(f"    {len(areas[area_name]):,} points")

    train_area_names = ['L001', 'L002', 'L004']
    test_area_name = 'L003'
    train_dfs = [areas[a] for a in train_area_names if a in areas]
    test_df = areas.get(test_area_name)

    if not train_dfs or test_df is None:
        print("ERROR: training or test files missing!")
        return

    print(f"\nTraining: {' + '.join(a for a in train_area_names if a in areas)}"
          f" ({sum(len(d) for d in train_dfs):,} points)")
    print(f"Test: {test_area_name} ({len(test_df):,} points)")

    # ── Miglioramenti v2 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("V2 IMPROVEMENTS")
    print("=" * 60)
    print("  [1] Data augmentation: rotazione Z, jitter, scaling")
    print("  [2] CrossEntropyLoss pesata (inverse sqrt) per class imbalance")
    print("  [3] CosineAnnealingWarmRestarts + Early Stopping (patience=15)")
    print("  [4] Blocchi 20×20 m (stride 10 m), ≥4096 points")
    print("  [5] AMP mixed precision + AdamW")

    # ── GridSearch ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GRID SEARCH v2")
    print("=" * 60)

    param_grid = {
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [32, 64],
        'num_points': [4096, 8192],
    }

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"Combinations: {n_combos}")
    print(f"Epochs per combination: {args.gs_epochs}")
    print(f"Blocchi: 20×20 m, stride 10 m")
    print(f"Loss: CrossEntropyLoss pesata\n")

    best_params, grid_results = grid_search(
        train_dfs, param_grid, device,
        num_epochs=args.gs_epochs, num_workers=args.num_workers,
    )

    results_df = pd.DataFrame(grid_results)
    print("\nGridSearch v2 results (sorted by mIoU):")
    print(results_df.sort_values('val_miou', ascending=False).to_string(index=False))
    results_df.to_csv(
        os.path.join(args.results_dir, 'gridsearch_v2_results.csv'), index=False,
    )
    # Save GS checkpoint
    with open(os.path.join(args.results_dir, '_checkpoint.json'), 'w') as f:
        json.dump({'stage': 'gridsearch_done', 'best_params': best_params}, f)
    print(f"[CHECKPOINT] GridSearch completed — best_params saved")

    # ── Cross-Validation ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{args.k_folds}-FOLD CROSS-VALIDATION v2")
    print("=" * 60)
    print(f"Best parameters: {best_params}")
    print(f"Max epochs: {args.cv_epochs} (with early stopping, patience=15)")

    cv_results = cross_validate(
        train_dfs, best_params, device,
        k=args.k_folds, num_epochs=args.cv_epochs,
        num_workers=args.num_workers, save_dir=args.results_dir,
    )

    mious = [r['best_miou'] for r in cv_results if r['best_miou'] > 0]
    epochs_trained = [r['epochs_trained'] for r in cv_results if r['epochs_trained'] > 0]
    print(f"\nCV Summary v2 — Mean mIoU: {np.mean(mious):.4f} ± {np.std(mious):.4f}")
    print(f"Avg epochs per fold: {np.mean(epochs_trained):.0f}")

    try:
        plot_cv_learning_curves(
            [r for r in cv_results if r['epochs_trained'] > 0],
            save_path=os.path.join(args.results_dir, 'cv_learning_curves_v2.png'),
        )
    except Exception as e:
        print(f"[WARNING] CV learning curves plot failed: {e}")

    # Save CV checkpoint
    with open(os.path.join(args.results_dir, '_checkpoint.json'), 'w') as f:
        json.dump({
            'stage': 'cv_done', 'best_params': best_params,
            'cv_mean_miou': float(np.mean(mious)),
            'cv_std_miou': float(np.std(mious)),
        }, f)
    print(f"[CHECKPOINT] Cross-Validation completed")

    # ── Final training ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL TRAINING v2")
    print("=" * 60)

    model_path = os.path.join(args.results_dir, 'best_model_v2_final.pth')
    final_model, train_dataset, criterion = train_final_model(
        train_dfs, best_params, device,
        num_epochs=args.final_epochs, num_workers=args.num_workers,
        save_path=model_path,
    )

    # ── Test set evaluation ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (L003) — v2")
    print("=" * 60)

    try:
        test_dataset = Toronto3DDatasetV2(
            [test_df], num_points=best_params['num_points'],
            block_size=20.0, stride=10.0,
            augment=False,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=best_params['batch_size'],
            shuffle=False, num_workers=args.num_workers,
            pin_memory=True,
        )
        print(f"Test set blocks: {len(test_dataset)}")

        test_loss, test_acc, test_miou, test_ious, test_preds, test_labels = evaluate(
            final_model, test_loader, criterion, device,
        )

        print_test_results(test_acc, test_miou, test_ious, test_labels, test_preds)

        # ── Plot ─────────────────────────────────────────────────────
        print("\nGenerating v2 plots...")

        plot_confusion_matrix(
            test_labels, test_preds,
            save_path=os.path.join(args.results_dir, 'confusion_matrix_v2.png'),
        )
        plot_iou_per_class(
            test_ious, test_miou,
            save_path=os.path.join(args.results_dir, 'iou_per_class_v2.png'),
        )

        sample_points, sample_labels = test_dataset[0]
        sample_dev = sample_points.unsqueeze(0).to(device)
        final_model.eval()
        with torch.no_grad():
            pred, _ = final_model(sample_dev)
            pred_np = pred.argmax(dim=-1).cpu().numpy().flatten()

        plot_prediction_comparison(
            sample_points.numpy(), sample_labels.numpy(), pred_np,
            save_path=os.path.join(args.results_dir, 'prediction_comparison_v2.png'),
        )

        # ── V1 vs V2 comparison ───────────────────────────────────────
        print("\n" + "=" * 60)
        print("V1 vs V2 COMPARISON")
        print("=" * 60)

        v1_results_path = os.path.join('results', 'gridsearch_results.csv')
        if os.path.exists(v1_results_path):
            v1_best_miou = pd.read_csv(v1_results_path)['val_miou'].max()
            print(f"  v1 — GS best mIoU:   {v1_best_miou:.4f}")
        print(f"  v2 — GS best mIoU:   {max(r['val_miou'] for r in grid_results):.4f}")
        print()
        print(f"  v2 — CV mIoU:         {np.mean(mious):.4f} ± {np.std(mious):.4f}")
        print(f"  v2 — Test Accuracy:   {test_acc:.4f}")
        print(f"  v2 — Test mIoU:       {test_miou:.4f}")

        # ── Summary ────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SUMMARY v2")
        print("=" * 60)
        print(f"  Overall Accuracy: {test_acc:.4f}")
        print(f"  Mean IoU:         {test_miou:.4f}")
        print(f"  CV mIoU:          {np.mean(mious):.4f} ± {np.std(mious):.4f}")
        print(f"  Model saved to:   {model_path}")
        print(f"  Results in:       {args.results_dir}/")

    except Exception as e:
        print(f"\n[ERROR] Test/plot failed: {e}")
        traceback.print_exc()
        print("The final model is still saved at:", model_path)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"PIPELINE CRASH: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        print("\nCheck results_v2/_checkpoint.json for reached state.")
        print("Partial results (models, CSV) are saved in results_v2/.")
        sys.exit(1)
