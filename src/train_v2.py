"""Training v2 — Cosine Annealing, Early Stopping, Mixed Precision.

Miglioramenti rispetto a v1:
  1. CrossEntropyLoss pesata (inverse sqrt) per class imbalance
  2. CosineAnnealingWarmRestarts scheduler
  3. Early stopping su mIoU (patience=15)
  4. AMP (Automatic Mixed Precision) per 2× throughput sulle L40S
  5. AdamW con weight decay
  6. pin_memory + più workers per I/O ottimale
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm

from .model import PointNetSegmentation, pointnet_regularization_loss
from .dataset import NUM_CLASSES
from .dataset_v2 import Toronto3DDatasetV2


# ── Focal Loss ───────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017).

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Riduce il contributo degli esempi facili e concentra il training
    sugli hard examples — efficace con class imbalance estremo.
    """

    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight, reduction='none',
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ── Early Stopping ───────────────────────────────────────────────

class EarlyStopping:
    """Early stopping che monitora mIoU (maximize)."""

    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0


# ── Utilità ──────────────────────────────────────────────────────

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _wrap_model(model):
    if torch.cuda.device_count() > 1:
        print(f"  Usando {torch.cuda.device_count()} GPU con DataParallel")
        model = nn.DataParallel(model)
    return model


def compute_class_weights(dataset, indices=None, num_classes=NUM_CLASSES):
    if indices is not None:
        all_labels = np.concatenate([dataset.labels[i] for i in indices])
    else:
        all_labels = np.concatenate(dataset.labels)
    counts = np.bincount(all_labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    # Inverse square root — meno aggressivo di 1/count, più stabile
    weights = 1.0 / np.sqrt(counts)
    weights /= weights.sum()
    return weights


# ── Training / Evaluation ────────────────────────────────────────

def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    scaler, reg_weight=0.001):
    """Epoca di training con AMP (mixed precision) e gradient clipping."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for points, labels in tqdm(dataloader, desc='Training', leave=False):
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda'):
            predictions, feat_t = model(points)
            predictions_flat = predictions.reshape(-1, predictions.size(-1))
            labels_flat = labels.reshape(-1)
            loss = criterion(predictions_flat, labels_flat)
            loss = loss + reg_weight * pointnet_regularization_loss(feat_t)

        # Controlla NaN/Inf — salta il batch se loss diverge
        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        # Gradient clipping per stabilità con Focal Loss + AMP
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        with torch.no_grad():
            _, predicted = predictions_flat.max(1)
            correct += predicted.eq(labels_flat).sum().item()
            total += labels_flat.size(0)

    return total_loss / max(len(dataloader), 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes=NUM_CLASSES):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for points, labels in tqdm(dataloader, desc='Evaluating', leave=False):
        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type='cuda'):
            predictions, feat_t = model(points)
            predictions_flat = predictions.reshape(-1, predictions.size(-1))
            labels_flat = labels.reshape(-1)
            loss = criterion(predictions_flat, labels_flat)

        total_loss += loss.item()
        _, predicted = predictions_flat.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels_flat.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    ious = []
    for cls in range(num_classes):
        intersection = ((all_preds == cls) & (all_labels == cls)).sum()
        union = ((all_preds == cls) | (all_labels == cls)).sum()
        ious.append(intersection / union if union > 0 else float('nan'))

    miou = np.nanmean(ious)
    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss, accuracy, miou, ious, all_preds, all_labels


# ── GridSearch ───────────────────────────────────────────────────

def grid_search(train_dfs, param_grid, device, num_epochs=10, num_workers=8):
    """GridSearch v2 con Focal Loss e AMP."""
    best_params = None
    best_val_miou = 0.0
    results = []
    scaler = GradScaler()

    for params in ParameterGrid(param_grid):
        print(f"\nTesting: {params}")

        try:
            dataset = Toronto3DDatasetV2(
                train_dfs,
                num_points=params['num_points'],
                block_size=20.0, stride=10.0,
                augment=True,
            )

            val_size = int(0.2 * len(dataset))
            train_size = len(dataset) - val_size
            train_subset, val_subset = torch.utils.data.random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

            train_loader = DataLoader(
                train_subset, batch_size=params['batch_size'],
                shuffle=True, num_workers=num_workers,
                pin_memory=True, drop_last=True,
            )
            val_loader = DataLoader(
                val_subset, batch_size=params['batch_size'],
                shuffle=False, num_workers=num_workers,
                pin_memory=True,
            )

            model = PointNetSegmentation(num_features=7, num_classes=NUM_CLASSES).to(device)
            model = _wrap_model(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=params['learning_rate'], weight_decay=1e-4,
            )

            weights = compute_class_weights(dataset, train_subset.indices)
            criterion = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(weights).to(device),
            )

            # Disattiva augmentation per la validazione GS
            for epoch in range(num_epochs):
                dataset.augment = True
                train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)

            dataset.augment = False
            val_loss, val_acc, val_miou, _, _, _ = evaluate(
                model, val_loader, criterion, device,
            )

            result = {**params, 'val_acc': val_acc, 'val_miou': val_miou, 'val_loss': val_loss}
            results.append(result)
            print(f"  Val Acc: {val_acc:.4f}, Val mIoU: {val_miou:.4f}")

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_params = params

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  OOM con {params} — skip")
                torch.cuda.empty_cache()
                results.append({**params, 'val_acc': 0, 'val_miou': 0, 'val_loss': float('inf')})
                continue
            raise

    print(f"\nBest params: {best_params} (mIoU: {best_val_miou:.4f})")
    return best_params, results


# ── Cross-Validation ─────────────────────────────────────────────

def cross_validate(train_dfs, params, device, k=5, num_epochs=100,
                   num_workers=8, save_dir='results_v2'):
    """CV v2 con Focal Loss, CosineAnnealingWarmRestarts, Early Stopping, AMP."""
    os.makedirs(save_dir, exist_ok=True)
    scaler = GradScaler()

    dataset = Toronto3DDatasetV2(
        train_dfs,
        num_points=params['num_points'],
        block_size=20.0, stride=10.0,
        augment=False,  # toggled per-epoch
    )

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)))):
        print(f"\n--- Fold {fold + 1}/{k} ---")

        try:
            train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
            val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())

            train_loader = DataLoader(
                train_subset, batch_size=params['batch_size'],
                shuffle=True, num_workers=num_workers,
                pin_memory=True, drop_last=True,
            )
            val_loader = DataLoader(
                val_subset, batch_size=params['batch_size'],
                shuffle=False, num_workers=num_workers,
                pin_memory=True,
            )

            model = PointNetSegmentation(num_features=7, num_classes=NUM_CLASSES).to(device)
            model = _wrap_model(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=params['learning_rate'], weight_decay=1e-4,
            )
            # Cosine annealing: T_0=20, poi 40, poi 80 (warm restarts)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=1e-6,
            )

            weights = compute_class_weights(dataset, train_idx.tolist())
            criterion = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(weights).to(device),
            )

            early_stop = EarlyStopping(patience=15)
            best_fold_miou = 0.0
            train_losses, val_losses = [], []

            for epoch in range(num_epochs):
                dataset.augment = True
                train_loss, _ = train_one_epoch(
                    model, train_loader, optimizer, criterion, device, scaler,
                )
                dataset.augment = False
                val_loss, val_acc, val_miou, _, _, _ = evaluate(
                    model, val_loader, criterion, device,
                )
                scheduler.step()

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if val_miou > best_fold_miou:
                    best_fold_miou = val_miou
                    raw = model.module if hasattr(model, 'module') else model
                    torch.save(
                        raw.state_dict(),
                        os.path.join(save_dir, f'best_model_v2_fold{fold}.pth'),
                    )

                if (epoch + 1) % 10 == 0:
                    print(
                        f"  Epoch {epoch + 1}: Train Loss={train_loss:.4f}, "
                        f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
                        f"Val mIoU={val_miou:.4f}, "
                        f"LR={optimizer.param_groups[0]['lr']:.2e}"
                    )

                early_stop(val_miou)
                if early_stop.should_stop:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

            cv_results.append({
                'fold': fold + 1,
                'best_miou': best_fold_miou,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'epochs_trained': len(train_losses),
            })
            print(f"  Best mIoU fold {fold + 1}: {best_fold_miou:.4f} "
                  f"({len(train_losses)} epochs)")

        except RuntimeError as e:
            print(f"  ERRORE nel fold {fold + 1}: {e}")
            torch.cuda.empty_cache()
            cv_results.append({
                'fold': fold + 1, 'best_miou': 0.0,
                'train_losses': [], 'val_losses': [],
                'epochs_trained': 0,
            })
            continue

    mious = [r['best_miou'] for r in cv_results if r['best_miou'] > 0]
    if mious:
        print(f"\nCV Results — Mean mIoU: {np.mean(mious):.4f} ± {np.std(mious):.4f}")
    else:
        print("\nCV Results — Tutti i fold falliti!")
    return cv_results


# ── Training finale ──────────────────────────────────────────────

def train_final_model(train_dfs, params, device, num_epochs=100, num_workers=8,
                      save_path='results_v2/best_model_v2_final.pth'):
    """Training finale v2 con tutti i miglioramenti."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scaler = GradScaler()

    dataset = Toronto3DDatasetV2(
        train_dfs,
        num_points=params['num_points'],
        block_size=20.0, stride=10.0,
        augment=True,
    )

    loader = DataLoader(
        dataset, batch_size=params['batch_size'],
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )

    model = PointNetSegmentation(num_features=7, num_classes=NUM_CLASSES).to(device)
    model = _wrap_model(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params['learning_rate'], weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6,
    )

    weights = compute_class_weights(dataset)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, loader, optimizer, criterion, device, scaler,
        )
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(
                f"  Epoch {epoch + 1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )

    raw = model.module if hasattr(model, 'module') else model
    torch.save(raw.state_dict(), save_path)
    print(f"Modello salvato in {save_path}")

    return model, dataset, criterion
