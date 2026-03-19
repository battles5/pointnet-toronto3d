"""Toronto-3D dataset loader and PyTorch Dataset for PointNet segmentation."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from plyfile import PlyData


# Mapping classi Toronto-3D
CLASS_NAMES = {
    0: 'Unclassified',
    1: 'Road',
    2: 'Road Marking',
    3: 'Natural',
    4: 'Building',
    5: 'Utility Line',
    6: 'Pole',
    7: 'Car',
    8: 'Fence',
}

NUM_CLASSES = 9


def _resolve_field(names, candidates):
    """Ritorna il primo campo presente tra i candidati, o None."""
    for c in candidates:
        if c in names:
            return c
    return None


# Offset UTM consigliato dal paper per evitare perdita di precisione
UTM_OFFSET = np.array([627285.0, 4841948.0, 0.0])


def load_toronto3d_ply(filepath):
    """Carica un file PLY del dataset Toronto-3D e ritorna un DataFrame.

    Il DataFrame contiene: x, y, z, r, g, b, intensity, gps_time, label.
    Le coordinate UTM vengono riscalate sottraendo UTM_OFFSET.
    """
    plydata = PlyData.read(filepath)
    vertex = plydata['vertex']
    names = vertex.data.dtype.names

    # Coordinate — sottrai offset UTM per mantenere la precisione float
    data = {
        'x': np.array(vertex['x'], dtype=np.float64) - UTM_OFFSET[0],
        'y': np.array(vertex['y'], dtype=np.float64) - UTM_OFFSET[1],
        'z': np.array(vertex['z'], dtype=np.float64) - UTM_OFFSET[2],
        'r': np.array(vertex['red'], dtype=np.float32),
        'g': np.array(vertex['green'], dtype=np.float32),
        'b': np.array(vertex['blue'], dtype=np.float32),
    }

    # Intensity — il nome cambia tra versioni del file
    int_field = _resolve_field(names, ('intensity', 'scalar_Intensity'))
    if int_field:
        data['intensity'] = np.array(vertex[int_field], dtype=np.float32)
    else:
        data['intensity'] = np.zeros(len(vertex.data), dtype=np.float32)

    # GPS time
    gps_field = _resolve_field(names, ('gps_time', 'scalar_GPSTime'))
    if gps_field:
        data['gps_time'] = np.array(vertex[gps_field], dtype=np.float64)

    # Label
    label_field = _resolve_field(names, ('scalar_Label', 'label', 'classification'))
    if label_field is None:
        raise ValueError(
            f"Campo label non trovato nel file PLY. Campi disponibili: {names}"
        )
    data['label'] = np.array(vertex[label_field], dtype=np.int64)

    return pd.DataFrame(data)


class Toronto3DDataset(Dataset):
    """Dataset per PointNet: suddivide le point cloud in blocchi spaziali
    e campiona un numero fisso di punti per blocco."""

    def __init__(
        self,
        dataframes,
        num_points=4096,
        num_classes=NUM_CLASSES,
        features=None,
        block_size=10.0,
        stride=5.0,
        normalize=True,
    ):
        """
        Args:
            dataframes: lista di DataFrame (una per area).
            num_points: punti per blocco.
            num_classes: numero di classi.
            features: colonne da usare come input (default: x,y,z,intensity,r,g,b).
            block_size: dimensione del blocco spaziale in metri.
            stride: passo tra blocchi.
            normalize: se normalizzare coordinate, intensity e RGB.
        """
        if features is None:
            features = ['x', 'y', 'z', 'intensity', 'r', 'g', 'b']

        self.num_points = num_points
        self.num_classes = num_classes
        self.features = features
        self.blocks = []
        self.labels = []

        for df in dataframes:
            self._create_blocks(df, block_size, stride, normalize)

    def _create_blocks(self, df, block_size, stride, normalize):
        """Suddivide la point cloud in blocchi spaziali."""
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
                    # Normalizza coordinate rispetto al centro del blocco
                    points[:, 0] -= (x_start + block_size / 2)
                    points[:, 1] -= (y_start + block_size / 2)
                    points[:, 2] -= points[:, 2].mean()
                    # Normalizza intensity in [0, 1]
                    if 'intensity' in self.features:
                        idx = self.features.index('intensity')
                        max_val = points[:, idx].max()
                        if max_val > 0:
                            points[:, idx] /= max_val
                    # Normalizza RGB in [0, 1]
                    for ch in ('r', 'g', 'b'):
                        if ch in self.features:
                            idx = self.features.index(ch)
                            if points[:, idx].max() > 1.0:
                                points[:, idx] /= 255.0

                self.blocks.append(points)
                self.labels.append(labels)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        points = self.blocks[idx]
        labels = self.labels[idx]

        # Campiona o padda a num_points
        n = len(points)
        if n >= self.num_points:
            choice = np.random.choice(n, self.num_points, replace=False)
        else:
            choice = np.random.choice(n, self.num_points, replace=True)

        points = points[choice]
        labels = labels[choice]

        return torch.FloatTensor(points), torch.LongTensor(labels)
