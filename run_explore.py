#!/usr/bin/env python3
"""Data exploration and visualization of the Toronto-3D dataset.

Generates all EDA plots and saves them to results/.
"""

import os
import argparse
import numpy as np

from src.dataset import load_toronto3d_ply, CLASS_NAMES
from src.utils import (
    plot_class_distribution,
    plot_pointcloud_3d,
    plot_birdseye_view,
    plot_feature_analysis,
    plot_correlation_matrix,
)


def main():
    parser = argparse.ArgumentParser(description='Toronto-3D data exploration')
    parser.add_argument('--data-dir', default='data/toronto3d',
                        help='Folder containing PLY files')
    parser.add_argument('--results-dir', default='results',
                        help='Output folder for plots')
    parser.add_argument('--max-points', type=int, default=100_000,
                        help='Max points for visualization')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Loading ──────────────────────────────────────────────
    areas = {}
    for area_name in ['L001', 'L002', 'L003', 'L004']:
        filepath = os.path.join(args.data_dir, f'{area_name}.ply')
        if os.path.exists(filepath):
            print(f"Loading {area_name}...")
            areas[area_name] = load_toronto3d_ply(filepath)
            print(f"  {area_name}: {len(areas[area_name]):,} points, "
                  f"columns: {list(areas[area_name].columns)}")
        else:
            print(f"  ⚠ {area_name}: file not found")

    if not areas:
        print("No PLY files found! Download from: "
              "https://github.com/WeikaiTan/Toronto-3D")
        return

    total = sum(len(df) for df in areas.values())
    print(f"\nTotal points loaded: {total:,}")

    # Use the first area for visualizations
    first_name = next(iter(areas))
    df = areas[first_name]

    # ── Descriptive statistics ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Area {first_name} statistics")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nDescriptive statistics:\n{df.describe()}")

    # ── Class distribution (all areas) ─────────────────────────────
    import pandas as pd
    all_df = pd.concat(areas.values(), ignore_index=True)

    print(f"\n{'='*60}")
    print("Class distribution across the entire dataset")
    print(f"{'='*60}")
    for cls_id, name in CLASS_NAMES.items():
        count = (all_df['label'] == cls_id).sum()
        pct = 100.0 * count / len(all_df)
        print(f"  {name:15s}: {count:>10,} ({pct:5.1f}%)")

    plot_class_distribution(
        all_df, area_name='(tutte le aree)',
        save_path=os.path.join(args.results_dir, 'class_distribution.png'),
    )

    # ── 3D visualization ───────────────────────────────────────
    plot_pointcloud_3d(
        df, color_by='label', max_points=args.max_points,
        save_path=os.path.join(args.results_dir, 'pointcloud_3d_classes.png'),
    )

    # ── Bird's eye view ──────────────────────────────────────────
    plot_birdseye_view(
        df, max_points=args.max_points,
        save_path=os.path.join(args.results_dir, 'birdseye_view.png'),
    )

    # ── Feature analysis ─────────────────────────────────────────
    plot_feature_analysis(
        df, max_points=50_000,
        save_path=os.path.join(args.results_dir, 'feature_analysis.png'),
    )

    # ── Correlation matrix ──────────────────────────────────
    plot_correlation_matrix(
        df,
        save_path=os.path.join(args.results_dir, 'correlation_matrix.png'),
    )

    print(f"\nPlots saved to {args.results_dir}/")


if __name__ == '__main__':
    main()
