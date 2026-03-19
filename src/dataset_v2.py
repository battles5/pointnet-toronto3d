"""Toronto-3D Dataset v2 — blocchi più grandi + data augmentation on-the-fly.

Miglioramenti rispetto a v1:
  1. Blocchi spaziali 20×20 m (stride 10 m) per più contesto
  2. Data augmentation: rotazione Z, jitter, scaling
  3. Riusa il loader PLY e le costanti da dataset.py
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import load_toronto3d_ply, CLASS_NAMES, NUM_CLASSES  # noqa: F401


class Toronto3DDatasetV2(Dataset):
    """Dataset PointNet v2 con data augmentation e blocchi spaziali più grandi."""

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
        """Data augmentation geometrica on-the-fly.

        1. Rotazione casuale attorno all'asse Z (0–2π)
        2. Jitter gaussiano su XYZ (σ=0.01, clip ±0.05)
        3. Scaling uniforme casuale (0.9–1.1)
        """
        pts = points.copy()

        # Rotazione casuale attorno a Z
        theta = np.random.uniform(0, 2 * np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
        pts[:, :2] = pts[:, :2] @ rotation.T

        # Jitter gaussiano su XYZ
        jitter = np.clip(
            np.random.normal(0, 0.01, size=(pts.shape[0], 3)).astype(np.float32),
            -0.05, 0.05,
        )
        pts[:, :3] += jitter

        # Scaling uniforme
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
