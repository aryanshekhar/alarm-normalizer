"""
SIMBA Training Pipeline
========================
Trains the SIMBA model on synthetic or real 5G KPI datasets.

Features:
  - Weighted/focal loss for class imbalance
  - Early stopping with patience
  - Model checkpointing (best val F1)
  - Per-class metrics (Precision, Recall, F1)
  - Learning rate scheduling (ReduceLROnPlateau)
  - TensorBoard logging (optional)

Usage:
    # Generate data first:
    python data/dataset_generator.py --duration 7200 --output data/kpi_dataset.npz

    # Train:
    python training/train.py --data data/kpi_dataset.npz --epochs 50 --output models/simba_best.pt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import json
from typing import Dict, Tuple, Optional
from datetime import datetime

from models.simba import Simba, WeightedFocalLoss, compute_class_weights


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 3,
    class_names: list = None,
) -> Dict:
    """Compute per-class and macro-averaged precision, recall, F1."""
    if class_names is None:
        class_names = ["normal", "power_reduction", "interference"]

    metrics = {}
    precisions, recalls, f1s = [], [], []

    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[f"precision_{class_names[c]}"] = float(precision)
        metrics[f"recall_{class_names[c]}"]    = float(recall)
        metrics[f"f1_{class_names[c]}"]        = float(f1)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    metrics["macro_precision"] = float(np.mean(precisions))
    metrics["macro_recall"]    = float(np.mean(recalls))
    metrics["macro_f1"]        = float(np.mean(f1s))
    metrics["accuracy"]        = float((y_pred == y_true).mean())

    # Anomaly detection metrics (binary: normal vs any fault)
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    tp_b = ((y_pred_bin == 1) & (y_true_bin == 1)).sum()
    fp_b = ((y_pred_bin == 1) & (y_true_bin == 0)).sum()
    fn_b = ((y_pred_bin == 0) & (y_true_bin == 1)).sum()
    prec_b = tp_b / (tp_b + fp_b + 1e-8)
    rec_b  = tp_b / (tp_b + fn_b + 1e-8)
    metrics["anomaly_precision"] = float(prec_b)
    metrics["anomaly_recall"]    = float(rec_b)
    metrics["anomaly_f1"]        = float(2 * prec_b * rec_b / (prec_b + rec_b + 1e-8))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model:     Simba,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    prior:     Optional[torch.Tensor] = None,
) -> Tuple[float, Dict]:
    """Run one evaluation pass. Returns (avg_loss, metrics_dict)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pr_b    = prior.to(device) if prior is not None else None

            logits, _ = model(X_batch, pr_b)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1).cpu().numpy()  # (B, N)
            all_preds.append(preds.flatten())
            all_labels.append(y_batch.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader)
    y_pred   = np.concatenate(all_preds)
    y_true   = np.concatenate(all_labels)
    metrics  = compute_metrics(y_true, y_pred)

    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience  = patience
        self.mode      = mode
        self.min_delta = min_delta
        self.best      = None
        self.counter   = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


def train_one_epoch(
    model:      Simba,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    prior:      Optional[torch.Tensor] = None,
    grad_clip:  float = 1.0,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pr_b    = prior.to(device) if prior is not None else None

        optimizer.zero_grad()
        logits, _ = model(X_batch, pr_b)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train(
    model:      Simba,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    config:       Dict,
    device:       torch.device,
    prior:        Optional[torch.Tensor] = None,
    output_path:  str = "models/simba_best.pt",
) -> Dict:
    """
    Full training loop with early stopping and LR scheduling.

    Returns:
        history : dict with train/val losses and metrics per epoch
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5,
        verbose=True, min_lr=1e-6,
    )
    early_stopper = EarlyStopping(
        patience=config.get("patience", 15), mode="max"
    )

    # Loss function — weighted focal loss for imbalance
    criterion = WeightedFocalLoss(
        n_classes=Simba.N_CLASSES,
        class_weights=config.get("class_weights"),
        gamma=config.get("focal_gamma", 2.0),
    ).to(device)

    history = {
        "train_loss": [], "val_loss": [],
        "val_macro_f1": [], "val_anomaly_f1": [],
        "val_accuracy": [],
    }
    best_val_f1   = 0.0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nTraining SIMBA on {device}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Epochs:     {config.get('epochs', 50)}")
    print(f"  Batch size: {config.get('batch_size', 32)}")
    print(f"  LR:         {config.get('lr', 1e-3)}")
    print()

    for epoch in range(1, config.get("epochs", 50) + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, prior
        )

        # Validate
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, prior
        )

        val_f1 = val_metrics["macro_f1"]
        scheduler.step(val_f1)

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        history["val_anomaly_f1"].append(val_metrics["anomaly_f1"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "val_f1":       val_f1,
                "val_metrics":  val_metrics,
                "config":       config,
            }, output_path)

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Macro-F1: {val_f1:.4f} | "
                f"Val Anomaly-F1: {val_metrics['anomaly_f1']:.4f} | "
                f"LR: {lr:.2e}"
            )

        # Early stopping
        if early_stopper.step(val_f1):
            print(f"\nEarly stopping at epoch {epoch} (best Val Macro-F1={best_val_f1:.4f})")
            break

    print(f"\nBest Val Macro-F1: {best_val_f1:.4f}")
    print(f"Model saved to {output_path}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SIMBA on 5G KPI dataset")
    parser.add_argument("--data",        type=str, default="data/kpi_dataset.npz")
    parser.add_argument("--output",      type=str, default="models/simba_best.pt")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int, default=15)
    parser.add_argument("--hidden-dim",  type=int, default=64)
    parser.add_argument("--n-heads",     type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--gcn-layers",  type=int, default=2)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--no-gpu",      action="store_true")
    args = parser.parse_args()

    # Device
    device = torch.device("cpu" if args.no_gpu or not torch.cuda.is_available()
                          else "cuda")

    # Load dataset
    print(f"Loading dataset from {args.data}")
    data = np.load(args.data, allow_pickle=True)
    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(data["y_train"], dtype=torch.long)
    X_val   = torch.tensor(data["X_val"],   dtype=torch.float32)
    y_val   = torch.tensor(data["y_val"],   dtype=torch.long)
    adj_np  = data["adjacency"]
    n_cells = int(data["n_cells"])
    n_kpis  = int(data["n_kpis"])
    window_size = X_train.shape[1]

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  n_cells={n_cells}, n_kpis={n_kpis}, window={window_size}")

    # DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=0)

    # Physical adjacency prior
    prior = torch.tensor(adj_np, dtype=torch.float32).to(device)

    # Class weights for imbalanced data
    class_weights = compute_class_weights(data["y_train"])
    print(f"  Class weights: {class_weights.numpy().round(3)}")

    # Build model
    model = Simba(
        n_kpis=n_kpis,
        n_cells=n_cells,
        window_size=window_size,
        gcn_hidden=args.hidden_dim,
        gcn_output=args.hidden_dim,
        gcn_layers=args.gcn_layers,
        temporal_dim=args.hidden_dim,
        n_heads=args.n_heads,
        transformer_layers=args.transformer_layers,
        ff_dim=args.hidden_dim * 2,
        fusion_hidden=args.hidden_dim * 2,
        dropout=args.dropout,
    ).to(device)

    print(f"\nModel architecture:")
    print(f"  Total parameters: {model.count_parameters():,}")

    # Training config
    config = {
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "patience":      args.patience,
        "weight_decay":  1e-4,
        "focal_gamma":   2.0,
        "class_weights": class_weights,
    }

    # Train
    history = train(
        model, train_loader, val_loader,
        config, device, prior, args.output
    )

    # Save history
    history_path = args.output.replace(".pt", "_history.json")
    history_serializable = {k: [float(v) for v in vals]
                            for k, vals in history.items()}
    with open(history_path, "w") as f:
        json.dump(history_serializable, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Final evaluation on test set
    print("\n--- Test Set Evaluation ---")
    X_test = torch.tensor(data["X_test"], dtype=torch.float32)
    y_test = torch.tensor(data["y_test"], dtype=torch.long)
    test_ds     = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Load best checkpoint
    checkpoint  = torch.load(args.output, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    criterion = WeightedFocalLoss(class_weights=class_weights).to(device)
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device, prior)

    print(f"Test Loss:          {test_loss:.4f}")
    print(f"Test Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"Test Macro-F1:      {test_metrics['macro_f1']:.4f}")
    print(f"Test Anomaly-F1:    {test_metrics['anomaly_f1']:.4f}")
    print(f"\nPer-class F1:")
    for name in ["normal", "power_reduction", "interference"]:
        print(f"  {name:25s}: {test_metrics.get('f1_'+name, 0):.4f}")


if __name__ == "__main__":
    main()
