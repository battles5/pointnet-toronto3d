"""Training, evaluation, GridSearch and Cross-Validation functions."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm

from .model import PointNetSegmentation, pointnet_regularization_loss
from .dataset import Toronto3DDataset, NUM_CLASSES


def get_device():
    """Return the available device (CUDA > CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _wrap_model(model):
    """Wrap the model with DataParallel if multiple GPUs are available."""
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    return model


def compute_class_weights(dataset, indices=None, num_classes=NUM_CLASSES):
    """Compute inverse weights to balance classes."""
    if indices is not None:
        all_labels = np.concatenate([dataset.labels[i] for i in indices])
    else:
        all_labels = np.concatenate(dataset.labels)
    counts = np.bincount(all_labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = 1.0 / counts
    weights /= weights.sum()
    return weights


def train_one_epoch(model, dataloader, optimizer, criterion, device, reg_weight=0.001):
    """Run one training epoch. Returns (avg loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for points, labels in tqdm(dataloader, desc='Training', leave=False):
        points, labels = points.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions, feat_t = model(points)

        predictions_flat = predictions.reshape(-1, predictions.size(-1))
        labels_flat = labels.reshape(-1)

        loss = criterion(predictions_flat, labels_flat)
        loss = loss + reg_weight * pointnet_regularization_loss(feat_t)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = predictions_flat.max(1)
        correct += predicted.eq(labels_flat).sum().item()
        total += labels_flat.size(0)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, num_classes=NUM_CLASSES):
    """Evaluate the model. Returns (loss, accuracy, mIoU, ious, preds, labels)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for points, labels in tqdm(dataloader, desc='Evaluating', leave=False):
        points, labels = points.to(device), labels.to(device)

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

    # Per-class IoU
    ious = []
    for cls in range(num_classes):
        intersection = ((all_preds == cls) & (all_labels == cls)).sum()
        union = ((all_preds == cls) | (all_labels == cls)).sum()
        ious.append(intersection / union if union > 0 else float('nan'))

    miou = np.nanmean(ious)
    avg_loss = total_loss / max(len(dataloader), 1)

    return avg_loss, accuracy, miou, ious, all_preds, all_labels


def grid_search(train_dfs, param_grid, device, num_epochs=5, num_workers=2):
    """Run GridSearch over hyperparameters.

    Args:
        train_dfs: list of training DataFrames.
        param_grid: dict with keys 'learning_rate', 'batch_size', 'num_points'.
        device: torch device.
        num_epochs: epochs per combination.
        num_workers: workers for DataLoader.

    Returns:
        best_params: dict with the best parameters.
        results: list of dicts with results.
    """
    best_params = None
    best_val_miou = 0.0
    results = []

    for params in ParameterGrid(param_grid):
        print(f"\nTesting: {params}")

        dataset = Toronto3DDataset(
            train_dfs,
            num_points=params['num_points'],
            block_size=10.0,
            stride=5.0,
        )

        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_subset, batch_size=params['batch_size'],
            shuffle=True, num_workers=num_workers, drop_last=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=params['batch_size'],
            shuffle=False, num_workers=num_workers,
        )

        model = PointNetSegmentation(num_features=7, num_classes=NUM_CLASSES).to(device)
        model = _wrap_model(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

        weights = compute_class_weights(dataset, train_subset.indices)
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(weights).to(device)
        )

        for epoch in range(num_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_loss, val_acc, val_miou, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        result = {**params, 'val_acc': val_acc, 'val_miou': val_miou, 'val_loss': val_loss}
        results.append(result)
        print(f"  Val Acc: {val_acc:.4f}, Val mIoU: {val_miou:.4f}")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_params = params

    print(f"\nBest params: {best_params} (mIoU: {best_val_miou:.4f})")
    return best_params, results


def cross_validate(train_dfs, params, device, k=5, num_epochs=30,
                   num_workers=2, save_dir='results'):
    """Run K-Fold Cross-Validation with the given parameters.

    Returns:
        cv_results: list of dicts per fold with 'fold', 'best_miou',
                    'train_losses', 'val_losses'.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    dataset = Toronto3DDataset(
        train_dfs,
        num_points=params['num_points'],
        block_size=10.0,
        stride=5.0,
    )

    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)))):
        print(f"\n--- Fold {fold + 1}/{k} ---")

        train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
        val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())

        train_loader = DataLoader(
            train_subset, batch_size=params['batch_size'],
            shuffle=True, num_workers=num_workers, drop_last=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=params['batch_size'],
            shuffle=False, num_workers=num_workers,
        )

        model = PointNetSegmentation(num_features=7, num_classes=NUM_CLASSES).to(device)
        model = _wrap_model(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        weights = compute_class_weights(dataset, train_idx.tolist())
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(weights).to(device)
        )

        best_fold_miou = 0.0
        train_losses, val_losses = [], []

        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, val_miou, _, _, _ = evaluate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_miou > best_fold_miou:
                best_fold_miou = val_miou
                raw = model.module if hasattr(model, 'module') else model
                torch.save(
                    raw.state_dict(),
                    os.path.join(save_dir, f'best_model_fold{fold}.pth'),
                )

            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch + 1}: Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, "
                    f"Val mIoU={val_miou:.4f}"
                )

        cv_results.append({
            'fold': fold + 1,
            'best_miou': best_fold_miou,
            'train_losses': train_losses,
            'val_losses': val_losses,
        })
        print(f"  Best mIoU fold {fold + 1}: {best_fold_miou:.4f}")

    mious = [r['best_miou'] for r in cv_results]
    print(f"\nCV Results — Mean mIoU: {np.mean(mious):.4f} ± {np.std(mious):.4f}")
    return cv_results


def train_final_model(train_dfs, params, device, num_epochs=30, num_workers=2,
                      save_path='results/best_model_final.pth'):
    """Train the final model on the entire training set."""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = Toronto3DDataset(
        train_dfs,
        num_points=params['num_points'],
        block_size=10.0,
        stride=5.0,
    )

    loader = DataLoader(
        dataset, batch_size=params['batch_size'],
        shuffle=True, num_workers=num_workers, drop_last=True,
    )

    model = PointNetSegmentation(num_features=7, num_classes=NUM_CLASSES).to(device)
    model = _wrap_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    weights = compute_class_weights(dataset)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, loader, optimizer, criterion, device
        )
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

    raw = model.module if hasattr(model, 'module') else model
    torch.save(raw.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model, dataset, criterion
