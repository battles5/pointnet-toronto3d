#!/usr/bin/env python3
"""Data exploration e visualizzazione del dataset Toronto-3D.

Genera tutti i plot di EDA e li salva in results/.
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
    parser = argparse.ArgumentParser(description='Esplorazione dati Toronto-3D')
    parser.add_argument('--data-dir', default='data/toronto3d',
                        help='Cartella con i file PLY')
    parser.add_argument('--results-dir', default='results',
                        help='Cartella di output per i plot')
    parser.add_argument('--max-points', type=int, default=100_000,
                        help='Punti massimi per visualizzazione')
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # ── Caricamento ──────────────────────────────────────────────
    areas = {}
    for area_name in ['L001', 'L002', 'L003', 'L004']:
        filepath = os.path.join(args.data_dir, f'{area_name}.ply')
        if os.path.exists(filepath):
            print(f"Caricamento {area_name}...")
            areas[area_name] = load_toronto3d_ply(filepath)
            print(f"  {area_name}: {len(areas[area_name]):,} punti, "
                  f"colonne: {list(areas[area_name].columns)}")
        else:
            print(f"  ⚠ {area_name}: file non trovato")

    if not areas:
        print("Nessun file PLY trovato! Scarica da: "
              "https://github.com/WeikaiTan/Toronto-3D")
        return

    total = sum(len(df) for df in areas.values())
    print(f"\nTotale punti caricati: {total:,}")

    # Usa la prima area per le visualizzazioni
    first_name = next(iter(areas))
    df = areas[first_name]

    # ── Statistiche descrittive ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Statistiche area {first_name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nTipi di dato:\n{df.dtypes}")
    print(f"\nStatistiche descrittive:\n{df.describe()}")

    # ── Distribuzione classi (tutte le aree) ─────────────────────
    import pandas as pd
    all_df = pd.concat(areas.values(), ignore_index=True)

    print(f"\n{'='*60}")
    print("Distribuzione classi su tutto il dataset")
    print(f"{'='*60}")
    for cls_id, name in CLASS_NAMES.items():
        count = (all_df['label'] == cls_id).sum()
        pct = 100.0 * count / len(all_df)
        print(f"  {name:15s}: {count:>10,} ({pct:5.1f}%)")

    plot_class_distribution(
        all_df, area_name='(tutte le aree)',
        save_path=os.path.join(args.results_dir, 'class_distribution.png'),
    )

    # ── Visualizzazione 3D ───────────────────────────────────────
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

    # ── Matrice di correlazione ──────────────────────────────────
    plot_correlation_matrix(
        df,
        save_path=os.path.join(args.results_dir, 'correlation_matrix.png'),
    )

    print(f"\nPlot salvati in {args.results_dir}/")


if __name__ == '__main__':
    main()
