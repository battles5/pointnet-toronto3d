"""Toronto-3D Dataset v2 — larger blocks + on-the-fly data augmentation.

Improvements over v1:
  1. Spatial blocks 20×20 m (stride 10 m) for more context
  2. Data augmentation: Z-rotation, jitter, scaling
  3. Reuses PLY loader and constants from dataset.py
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import load_toronto3d_ply, CLASS_NAMES, NUM_CLASSES  # noqa: F401


class Toronto3DDatasetV2(Dataset):
    """PointNet Dataset v2 with data augmentation and larger spatial blocks."""

    def __init__(
        self,
        dataframes,
        num_points=4096,
        num_classes=NUM_CLASSES,
        features=None,
        block_size=20.0,
        stride=10.0,
        normalize=True,
        augment=False,
    ):
        if features is None:
            features = ['x', 'y', 'z', 'intensity', 'r', 'g', 'b']

        self.num_points = num_points
        self.num_classes = num_classes
        self.features = features
        self.augment = augment
        self.blocks = []
        self.labels = []

        for df in dataframes:
            self._create_blocks(df, block_size, stride, normalize)

    def _create_blocks(self, df, block_size, stride, normalize):
        coords = df[['x', 'y']].values
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        x_range = np.arange(x_min, x_max - block_size, stride)
        y_range = np.arange(y_min, y_max - block_size, stride)

        for x_start in x_range:
            for y_start in y_range:
                mask = (
                    (df['x'] >= x_start) & (df['x'] < x_start + block_size)
                    & (df['y'] >= y_start) & (df['y'] < y_start + block_size)
                )
                block_df = df[mask]

                if len(block_df) < 100:
                    continue

                points = block_df[self.features].values.astype(np.float32)
                labels = block_df['label'].values.astype(np.int64)

                if normalize:
                    points[:, 0] -= (x_start + block_size / 2)
                    points[:, 1] -= (y_start + block_size / 2)
                    points[:, 2] -= points[:, 2].mean()
                    if 'intensity' in self.features:
                        idx = self.features.index('intensity')
                        max_val = points[:, idx].max()
                        if max_val > 0:
                            points[:, idx] /= max_val
                    for ch in ('r', 'g', 'b'):
                        if ch in self.features:
                            idx = self.features.index(ch)
                            if points[:, idx].max() > 1.0:
                                points[:, idx] /= 255.0

                self.blocks.append(points)
                self.labels.append(labels)

    @staticmethod
    def _augment_points(points):
        """Geometric data augmentation on-the-fly.

        1. Random rotation around the Z-axis (0–2π)
        2. Gaussian jitter on XYZ (σ=0.01, clip ±0.05)
        3. Random uniform scaling (0.9–1.1)
        """
        pts = points.copy()

        # Random rotation around Z
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        pts[:, :2] = pts[:, :2] @ rotation.T

        # Gaussian jitter on XYZ
        jitter = np.clip(
            np.random.normal(0, 0.01, size=(pts.shape[0], 3)).astype(np.float32),
            -0.05, 0.05,
        )
        pts[:, :3] += jitter

        # Uniform scaling
        scale = np.random.uniform(0.9, 1.1)
        pts[:, :3] *= scale

        return pts

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        points = self.blocks[idx]
        labels = self.labels[idx]

        n = len(points)
        if n >= self.num_points:
            choice = np.random.choice(n, self.num_points, replace=False)
        else:
            choice = np.random.choice(n, self.num_points, replace=True)

        points = points[choice]
        labels = labels[choice]

        if self.augment:
            points = self._augment_points(points)

        return torch.FloatTensor(points), torch.LongTensor(labels)
