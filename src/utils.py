"""Utility functions for visualization and results analysis."""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for SLURM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from .dataset import CLASS_NAMES, NUM_CLASSES


def plot_class_distribution(df, area_name='', save_path=None):
    """Plot class distribution with bar chart and pie chart."""
    class_counts = df['label'].value_counts().sort_index()
    class_counts.index = class_counts.index.map(CLASS_NAMES)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(class_counts.index, class_counts.values, color='steelblue')
    axes[0].set_xlabel('Number of points')
    axes[0].set_title(f'Class distribution {area_name}')
    for i, v in enumerate(class_counts.values):
        axes[0].text(v + 1000, i, f'{v:,}', va='center', fontsize=9)

    axes[1].pie(
        class_counts.values, labels=class_counts.index,
        autopct='%1.1f%%', startangle=140,
    )
    axes[1].set_title('Class proportion')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Most frequent class: {class_counts.idxmax()} ({class_counts.max():,})")
    print(f"Least frequent class: {class_counts.idxmin()} ({class_counts.min():,})")
    print(f"Max/min ratio: {class_counts.max() / class_counts.min():.1f}x")


def plot_pointcloud_3d(df, color_by='label', max_points=100000, save_path=None):
    """3D point cloud visualization."""
    if len(df) > max_points:
        sample_idx = np.random.choice(len(df), max_points, replace=False)
        df_sample = df.iloc[sample_idx]
    else:
        df_sample = df

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    if color_by == 'label':
        cmap = plt.cm.get_cmap('tab10', NUM_CLASSES)
        colors = cmap(df_sample['label'].values / (NUM_CLASSES - 1))
        ax.scatter(
            df_sample['x'], df_sample['y'], df_sample['z'],
            c=colors[:, :3], s=0.1, alpha=0.5,
        )
        ax.set_title('Point Cloud - Colored by class')
    elif color_by == 'rgb':
        rgb = df_sample[['r', 'g', 'b']].values
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        ax.scatter(
            df_sample['x'], df_sample['y'], df_sample['z'],
            c=rgb, s=0.1, alpha=0.5,
        )
        ax.set_title('Point Cloud - RGB color')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_birdseye_view(df, max_points=100000, save_path=None):
    """Bird's-eye view: RGB vs semantic classes."""
    if len(df) > max_points:
        idx = np.random.choice(len(df), max_points, replace=False)
        df_s = df.iloc[idx]
    else:
        df_s = df

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # RGB
    rgb = df_s[['r', 'g', 'b']].values
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    axes[0].scatter(df_s['x'], df_s['y'], c=rgb, s=0.1, alpha=0.5)
    axes[0].set_title("Bird's-eye view - RGB")
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_aspect('equal')

    # Semantic classes
    cmap = plt.cm.get_cmap('tab10', NUM_CLASSES)
    colors = cmap(df_s['label'].values / (NUM_CLASSES - 1))
    axes[1].scatter(df_s['x'], df_s['y'], c=colors[:, :3], s=0.1, alpha=0.5)
    axes[1].set_title("Bird's-eye view - Semantic classes")
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_feature_analysis(df, max_points=50000, save_path=None):
    """Box plots of features per class + correlation matrix."""
    if len(df) > max_points:
        idx = np.random.choice(len(df), max_points, replace=False)
        df_s = df.iloc[idx].copy()
    else:
        df_s = df.copy()

    df_s['class_name'] = df_s['label'].map(CLASS_NAMES)
    features = ['z', 'intensity', 'r', 'g', 'b']

    fig, axes = plt.subplots(1, len(features), figsize=(20, 5))
    for i, feat in enumerate(features):
        sns.boxplot(data=df_s, x='class_name', y=feat, ax=axes[i])
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_title(f'{feat} per class')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_correlation_matrix(df, save_path=None):
    """Feature correlation matrix."""
    cols = ['x', 'y', 'z', 'intensity', 'r', 'g', 'b']
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Feature correlation matrix')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


def plot_confusion_matrix(true_labels, pred_labels, save_path=None):
    """Normalized confusion matrix."""
    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Blues',
        xticklabels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        yticklabels=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
    )
    plt.title('Confusion Matrix (normalized) - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


def plot_iou_per_class(ious, miou, save_path=None):
    """Bar chart of per-class IoU."""
    valid = [(CLASS_NAMES[i], iou) for i, iou in enumerate(ious) if not np.isnan(iou)]
    names, values = zip(*valid)
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))

    plt.figure(figsize=(10, 5))
    plt.bar(names, values, color=colors)
    plt.axhline(y=miou, color='red', linestyle='--', label=f'mIoU = {miou:.3f}')
    plt.ylabel('IoU')
    plt.title('Per-class IoU on test set')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close('all')


def plot_cv_learning_curves(cv_results, save_path=None):
    """Learning curves for each cross-validation fold."""
    k = len(cv_results)
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))
    if k == 1:
        axes = [axes]

    for i, r in enumerate(cv_results):
        axes[i].plot(r['train_losses'], label='Train')
        axes[i].plot(r['val_losses'], label='Val')
        axes[i].set_title(f"Fold {r['fold']} (mIoU={r['best_miou']:.3f})")
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_prediction_comparison(points_np, true_labels, pred_labels, save_path=None):
    """Visual comparison of ground truth vs prediction on a block."""
    cmap = plt.cm.get_cmap('tab10', NUM_CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, labels, title in zip(
        axes, [true_labels, pred_labels],
        ['Ground Truth', 'PointNet Prediction'],
    ):
        colors = cmap(labels / (NUM_CLASSES - 1))
        ax.scatter(points_np[:, 0], points_np[:, 1], c=colors[:, :3], s=1, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('X (normalized)')
        ax.set_ylabel('Y (normalized)')
        ax.set_aspect('equal')

    plt.suptitle('Ground Truth vs Prediction Comparison', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def print_test_results(test_acc, test_miou, test_ious, true_labels, pred_labels):
    """Print a complete summary of test set results."""
    print(f"Overall Accuracy: {test_acc:.4f}")
    print(f"Mean IoU: {test_miou:.4f}")
    print(f"\nPer-class IoU:")
    for cls, iou in enumerate(test_ious):
        name = CLASS_NAMES[cls]
        if np.isnan(iou):
            print(f"  {name:15s}: N/A")
        else:
            print(f"  {name:15s}: {iou:.4f}")

    report = classification_report(
        true_labels, pred_labels,
        target_names=[CLASS_NAMES[i] for i in range(NUM_CLASSES)],
        digits=4, zero_division=0,
    )
    print(f"\nClassification Report:\n{report}")
