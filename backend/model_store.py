"""
In-memory store for the most recently trained SIMBA model.
Shared between train_model (writer) and run_inference (reader).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch


@dataclass
class ModelState:
    model: object              # simba_pipeline.models.simba.Simba
    normalizer: object         # simba_pipeline.data.dataset_generator.KPINormalizer
    prior: torch.Tensor        # (N_CELLS, N_CELLS) physical adjacency prior
    adjacency: np.ndarray      # same value, kept as numpy for serialisation
    config: dict               # hyperparams + final val metrics + anomaly_threshold
    anomalous_window: Optional[np.ndarray] = None  # (W, N_CELLS, N_KPIS) raw KPI window
    trained_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    inference_timestamp: Optional[datetime] = None
    alarms_fired_timestamp: Optional[datetime] = None


_state: Optional[ModelState] = None


def store(
    model,
    normalizer,
    prior: torch.Tensor,
    adjacency: np.ndarray,
    config: dict,
    anomalous_window: Optional[np.ndarray] = None,
) -> None:
    global _state
    _state = ModelState(
        model=model,
        normalizer=normalizer,
        prior=prior,
        adjacency=adjacency,
        config=config,
        anomalous_window=anomalous_window,
    )


def load() -> Optional[ModelState]:
    return _state


def is_ready() -> bool:
    return _state is not None
