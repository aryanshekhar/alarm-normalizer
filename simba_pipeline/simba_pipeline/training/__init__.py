"""
Training module — model training, evaluation, and metrics.

Key exports:
    train               — full training loop with early stopping and LR scheduling
    evaluate            — single evaluation pass returning loss and metrics dict
    compute_metrics     — precision, recall, F1 per class and macro-averaged
    EarlyStopping       — configurable patience-based early stopping
"""
from training.train import (
    train,
    evaluate,
    compute_metrics,
    EarlyStopping,
)
