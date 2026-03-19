#!/usr/bin/env python3
"""Full pipeline: GridSearch → Cross-Validation → Final Training → Evaluation.

Runs the entire ML workflow and saves models, results and plots to results/.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset import load_toronto3d_ply, Toronto3DDataset, NUM_CLASSES
from src.model import PointNetSegmentation
from src.train import (
    get_device,
    grid_search,
    cross_validate,
    train_final_model,
    evaluate,
    compute_class_weights,
)
from src.utils import (
    plot_confusion_matrix,
    plot_iou_per_class,
    plot_cv_learning_curves,
    plot_prediction_comparison,
    print_test_results,
)


def main():
    parser = argparse.ArgumentParser(description='PointNet ML pipeline on Toronto-3D')
    parser.add_argument('--data-dir', default='data/toronto3d',
                        help='Folder containing PLY files')
    parser.add_argument('--results-dir', default='results',
                        help='Output folder')
    parser.add_argument('--gs-epochs', type=int, default=5,
                        help='Epochs for GridSearch')
    parser.add_argument('--cv-epochs', type=int, default=30,
                        help='Epochs for Cross-Validation')
    parser.add_argument('--final-epochs', type=int, default=30,
                        help='Epochs for final training')
    parser.add_argument('--k-folds', type=int, default=5,
                        help='Number of folds for CV')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Workers for DataLoader')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data loading ─────────────────────────────────────────
    print("\n" + "="*60)
    print("DATA LOADING")
    print("="*60)

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

    if not train_dfs:
        print("ERROR: no training files found!")
        return
    if test_df is None:
        print("ERROR: test file (L003) not found!")
        return

    print(f"\nTraining: {' + '.join(a for a in train_area_names if a in areas)}"
          f" ({sum(len(d) for d in train_dfs):,} points)")
    print(f"Test: {test_area_name} ({len(test_df):,} points)")

    # ── GridSearch ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("GRID SEARCH")
    print("="*60)

    param_grid = {
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [16, 32],
        'num_points': [2048, 4096],
    }

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"Combinations: {n_combos}")
    print(f"Epochs per combination: {args.gs_epochs}\n")

    best_params, grid_results = grid_search(
        train_dfs, param_grid, device,
        num_epochs=args.gs_epochs, num_workers=args.num_workers,
    )

    results_df = pd.DataFrame(grid_results)
    print("\nGridSearch results (sorted by mIoU):")
    print(results_df.sort_values('val_miou', ascending=False).to_string(index=False))
    results_df.to_csv(
        os.path.join(args.results_dir, 'gridsearch_results.csv'), index=False
    )

    # ── Cross-Validation ─────────────────────────────────────────
    print("\n" + "="*60)
    print(f"{args.k_folds}-FOLD CROSS-VALIDATION")
    print("="*60)
    print(f"Best parameters: {best_params}")

    cv_results = cross_validate(
        train_dfs, best_params, device,
        k=args.k_folds, num_epochs=args.cv_epochs,
        num_workers=args.num_workers, save_dir=args.results_dir,
    )

    mious = [r['best_miou'] for r in cv_results]
    print(f"\nCV Summary — Mean mIoU: {np.mean(mious):.4f} ± {np.std(mious):.4f}")

    plot_cv_learning_curves(
        cv_results,
        save_path=os.path.join(args.results_dir, 'cv_learning_curves.png'),
    )

    # ── Final training ──────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL TRAINING")
    print("="*60)

    model_path = os.path.join(args.results_dir, 'best_model_final.pth')
    final_model, train_dataset, criterion = train_final_model(
        train_dfs, best_params, device,
        num_epochs=args.final_epochs, num_workers=args.num_workers,
        save_path=model_path,
    )

    # ── Test set evaluation ─────────────────────────────────
    print("\n" + "="*60)
    print("TEST SET EVALUATION (L003)")
    print("="*60)

    test_dataset = Toronto3DDataset(
        [test_df], num_points=best_params['num_points'],
        block_size=10.0, stride=5.0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=best_params['batch_size'],
        shuffle=False, num_workers=args.num_workers,
    )
    print(f"Test set blocks: {len(test_dataset)}")

    test_loss, test_acc, test_miou, test_ious, test_preds, test_labels = evaluate(
        final_model, test_loader, criterion, device,
    )

    print_test_results(test_acc, test_miou, test_ious, test_labels, test_preds)

    # ── Plot dei risultati ───────────────────────────────────────
    print("\nGenerating result plots...")

    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=os.path.join(args.results_dir, 'confusion_matrix.png'),
    )
    plot_iou_per_class(
        test_ious, test_miou,
        save_path=os.path.join(args.results_dir, 'iou_per_class.png'),
    )

    # Prediction vs GT on a block
    sample_points, sample_labels = test_dataset[0]
    sample_dev = sample_points.unsqueeze(0).to(device)
    final_model.eval()
    with torch.no_grad():
        pred, _ = final_model(sample_dev)
        pred_np = pred.argmax(dim=-1).cpu().numpy().flatten()

    plot_prediction_comparison(
        sample_points.numpy(), sample_labels.numpy(), pred_np,
        save_path=os.path.join(args.results_dir, 'prediction_comparison.png'),
    )

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Overall Accuracy: {test_acc:.4f}")
    print(f"  Mean IoU:         {test_miou:.4f}")
    print(f"  CV mIoU:          {np.mean(mious):.4f} ± {np.std(mious):.4f}")
    print(f"  Modello salvato:  {model_path}")
    print(f"  Risultati in:     {args.results_dir}/")


if __name__ == '__main__':
    main()
