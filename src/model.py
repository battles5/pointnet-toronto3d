"""PointNet per segmentazione semantica di point cloud (PyTorch)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Transformation Network (T-Net).

    Impara una matrice di trasformazione affine kxk per allineare
    i punti (o le features) in uno spazio canonico.
    """

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # Global max pooling

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Inizializza come matrice identità
        identity = (
            torch.eye(self.k, device=x.device)
            .flatten()
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x


class PointNetSegmentation(nn.Module):
    """PointNet per segmentazione semantica di point cloud.

    Architettura:
    1. Input Transform (T-Net kxk)
    2. Shared MLP (64, 64)
    3. Feature Transform (T-Net 64x64)
    4. Shared MLP (64, 128, 1024)
    5. Global feature (max pooling)
    6. Concatenazione feature locali + globali
    7. MLP per classificazione per-punto
    """

    def __init__(self, num_features=7, num_classes=9):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        # Input transform
        self.input_transform = TNet(k=num_features)

        # Shared MLP 1
        self.conv1 = nn.Conv1d(num_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Feature transform
        self.feature_transform = TNet(k=64)

        # Shared MLP 2
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Segmentation head (1024 global + 64 local = 1088)
        self.seg_conv1 = nn.Conv1d(1088, 512, 1)
        self.seg_conv2 = nn.Conv1d(512, 256, 1)
        self.seg_conv3 = nn.Conv1d(256, 128, 1)
        self.seg_conv4 = nn.Conv1d(128, num_classes, 1)
        self.seg_bn1 = nn.BatchNorm1d(512)
        self.seg_bn2 = nn.BatchNorm1d(256)
        self.seg_bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Args:
            x: (batch, num_points, num_features)
        Returns:
            out: (batch, num_points, num_classes)
            feat_transform: matrice di trasformazione delle features
        """
        batch_size, num_points, _ = x.size()

        # (batch, features, points) per Conv1d
        x = x.transpose(2, 1)

        # Input transform
        input_t = self.input_transform(x)
        x = torch.bmm(input_t, x)

        # Shared MLP 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transform
        feat_t = self.feature_transform(x)
        x = torch.bmm(feat_t, x)
        local_features = x  # Salva per concatenazione

        # Shared MLP 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Global feature
        global_feature = torch.max(x, 2)[0]  # (batch, 1024)
        global_feature = global_feature.unsqueeze(2).repeat(1, 1, num_points)

        # Concatena local + global
        x = torch.cat([local_features, global_feature], dim=1)

        # Segmentation head
        x = F.relu(self.seg_bn1(self.seg_conv1(x)))
        x = F.relu(self.seg_bn2(self.seg_conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.seg_bn3(self.seg_conv3(x)))
        x = self.seg_conv4(x)

        x = x.transpose(2, 1)  # (batch, points, classes)

        return x, feat_t


def pointnet_regularization_loss(feat_transform):
    """Regularizzazione: la matrice di trasformazione delle features
    deve essere quasi ortogonale (A * A^T ≈ I)."""
    batch_size = feat_transform.size(0)
    k = feat_transform.size(1)
    identity = torch.eye(k, device=feat_transform.device).unsqueeze(0)
    diff = torch.bmm(feat_transform, feat_transform.transpose(2, 1)) - identity
    return torch.mean(torch.norm(diff, dim=(1, 2)))
