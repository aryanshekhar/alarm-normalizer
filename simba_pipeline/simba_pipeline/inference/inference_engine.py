"""
SIMBA Real-Time Inference Engine
==================================
Processes live KPI streams from a real telecom network.

Maintains a sliding window buffer per cell and runs inference
whenever a full window is available. Outputs anomaly detections
with cell ID, fault type, confidence, and recommended action.

Integration points:
  - Kafka consumer (streaming KPIs from NMS/OSS)
  - REST API endpoint (pull-based from PM collector)
  - Direct numpy array (batch offline inference)

Usage:
    from inference.inference_engine import SimbaInferenceEngine

    engine = SimbaInferenceEngine(
        model_path="models/simba_best.pt",
        normalizer_path="data/kpi_dataset_normalizer.npz",
        adjacency=adj_matrix,
    )

    # Feed one timestep of KPIs for all cells
    result = engine.ingest(kpi_vector)   # kpi_vector: (n_cells, n_kpis)
    if result:
        for detection in result.detections:
            print(detection)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from models.simba import Simba
from data.dataset_generator import KPINormalizer, KPI_NAMES, N_KPIS


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

FAULT_TYPE_NAMES = {
    0: "normal",
    1: "excessive_power_reduction",
    2: "interference",
}

REPAIR_ACTIONS = {
    "normal": "No action required.",
    "excessive_power_reduction": (
        "Check TX power configuration on affected cell. "
        "Verify RRH connection and hardware status. "
        "Review SON power control parameters."
    ),
    "interference": (
        "Activate interference cancellation if available. "
        "Check for rogue transmitters in the frequency band. "
        "Consider frequency refarming or ICIC activation."
    ),
}


@dataclass
class CellDetection:
    """Anomaly detection result for a single cell."""
    cell_id:       int
    gnb_id:        int
    fault_type:    str
    confidence:    float              # 0..1
    probabilities: Dict[str, float]   # per-class probabilities
    timestamp:     str
    repair_action: str
    is_anomaly:    bool

    def to_dict(self) -> Dict:
        return {
            "cell_id":       self.cell_id,
            "gnb_id":        self.gnb_id,
            "fault_type":    self.fault_type,
            "confidence":    round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "timestamp":     self.timestamp,
            "is_anomaly":    self.is_anomaly,
            "repair_action": self.repair_action,
        }

    def __str__(self) -> str:
        status = "ANOMALY" if self.is_anomaly else "NORMAL"
        return (
            f"[{self.timestamp}] Cell {self.cell_id:3d} (gNB {self.gnb_id}) | "
            f"{status:7s} | {self.fault_type:30s} | "
            f"conf={self.confidence:.3f}"
        )


@dataclass
class InferenceResult:
    """Inference result for one window across all cells."""
    timestamp:      str
    n_cells:        int
    n_anomalies:    int
    detections:     List[CellDetection]
    learned_adj:    Optional[np.ndarray] = None  # for visualisation

    @property
    def anomalous_cells(self) -> List[CellDetection]:
        return [d for d in self.detections if d.is_anomaly]

    def summary(self) -> str:
        if self.n_anomalies == 0:
            return f"[{self.timestamp}] All {self.n_cells} cells NORMAL"
        cell_ids = [str(d.cell_id) for d in self.anomalous_cells]
        return (
            f"[{self.timestamp}] ANOMALY DETECTED — "
            f"{self.n_anomalies}/{self.n_cells} cells affected: "
            f"cells [{', '.join(cell_ids)}]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sliding Window Buffer
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowBuffer:
    """
    Maintains a fixed-size sliding window buffer of KPI observations
    per cell. Thread-safe for single-producer single-consumer usage.
    """

    def __init__(self, n_cells: int, n_kpis: int, window_size: int):
        self.n_cells     = n_cells
        self.n_kpis      = n_kpis
        self.window_size = window_size
        # Deque with fixed max length auto-evicts oldest entries
        self.buffer      = deque(maxlen=window_size)
        self._n_ingested = 0

    def push(self, kpi_snapshot: np.ndarray) -> None:
        """
        Push one timestep of KPIs for all cells.
        kpi_snapshot: (n_cells, n_kpis)
        """
        assert kpi_snapshot.shape == (self.n_cells, self.n_kpis), (
            f"Expected ({self.n_cells}, {self.n_kpis}), got {kpi_snapshot.shape}"
        )
        self.buffer.append(kpi_snapshot.copy())
        self._n_ingested += 1

    @property
    def is_ready(self) -> bool:
        """True when the buffer has a full window."""
        return len(self.buffer) == self.window_size

    def get_window(self) -> np.ndarray:
        """
        Returns the current window as (window_size, n_cells, n_kpis).
        Only valid when is_ready is True.
        """
        return np.array(list(self.buffer), dtype=np.float32)

    @property
    def n_ingested(self) -> int:
        return self._n_ingested


# ─────────────────────────────────────────────────────────────────────────────
# Inference Engine
# ─────────────────────────────────────────────────────────────────────────────

class SimbaInferenceEngine:
    """
    Real-time SIMBA inference engine.

    Workflow:
      1. Ingest one timestep of KPI data (all cells)
      2. Push to sliding window buffer
      3. When buffer is full, run normalisation + model inference
      4. Return CellDetection results for all cells

    The engine runs inference every `stride` timesteps after the
    initial window is filled, not on every single timestep.
    """

    def __init__(
        self,
        model_path:      str,
        normalizer_path: str,
        adjacency:       np.ndarray,         # physical topology adj (n_cells, n_cells)
        cell_to_gnb_map: Optional[Dict[int, int]] = None,
        window_size:     int   = 30,
        stride:          int   = 1,          # run inference every N timesteps
        anomaly_threshold: float = 0.5,      # confidence threshold
        device:          str   = "auto",
    ):
        # Device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Load model
        print(f"Loading SIMBA model from {model_path}...")
        checkpoint   = torch.load(model_path, map_location=self._device)
        model_config = checkpoint.get("config", {})

        # Infer model dimensions from checkpoint
        state = checkpoint["model_state"]
        n_cells = adjacency.shape[0]
        # Infer n_kpis from input projection weight
        n_kpis  = state["gcn.input_proj.weight"].shape[1]

        self._model = Simba(
            n_kpis=n_kpis,
            n_cells=n_cells,
            window_size=window_size,
        ).to(self._device)
        self._model.load_state_dict(state)
        self._model.eval()
        print(f"  Model loaded. Parameters: {self._model.count_parameters():,}")

        # Load normaliser
        self._normalizer = KPINormalizer.load(normalizer_path)

        # Physical topology prior
        self._prior = torch.tensor(adjacency, dtype=torch.float32).to(self._device)

        # Sliding window buffer
        self._buffer = SlidingWindowBuffer(n_cells, n_kpis, window_size)

        # Config
        self._n_cells           = n_cells
        self._n_kpis            = n_kpis
        self._window_size       = window_size
        self._stride            = stride
        self._anomaly_threshold = anomaly_threshold
        self._cell_to_gnb_map   = cell_to_gnb_map or {i: i // 3 for i in range(n_cells)}
        self._steps_since_infer = 0
        self._total_inferences  = 0

    def ingest(
        self, kpi_snapshot: np.ndarray
    ) -> Optional[InferenceResult]:
        """
        Ingest one timestep of KPI data.

        Args:
            kpi_snapshot : np.ndarray (n_cells, n_kpis) — raw (unnormalised) KPIs

        Returns:
            InferenceResult if inference was triggered, else None
        """
        self._buffer.push(kpi_snapshot)
        self._steps_since_infer += 1

        if not self._buffer.is_ready:
            return None
        if self._steps_since_infer < self._stride:
            return None

        self._steps_since_infer = 0
        return self._run_inference()

    def _run_inference(self) -> InferenceResult:
        """Run model inference on the current window."""
        # Get window and normalise
        window_raw = self._buffer.get_window()  # (W, N, K)
        window_norm = self._normalizer.transform(window_raw)

        # To tensor: (1, W, N, K)
        x = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits, adj = self._model(x, self._prior)
            probs = F.softmax(logits, dim=-1)  # (1, N, 3)

        probs_np = probs.squeeze(0).cpu().numpy()   # (N, 3)
        adj_np   = adj.cpu().numpy()
        ts       = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        detections = []
        for cell_id in range(self._n_cells):
            cell_probs = probs_np[cell_id]           # (3,)
            pred_class = int(cell_probs.argmax())
            confidence = float(cell_probs[pred_class])
            fault_name = FAULT_TYPE_NAMES[pred_class]
            is_anomaly = (pred_class > 0) and (confidence >= self._anomaly_threshold)

            detections.append(CellDetection(
                cell_id    = cell_id,
                gnb_id     = self._cell_to_gnb_map.get(cell_id, cell_id // 3),
                fault_type = fault_name,
                confidence = confidence,
                probabilities = {
                    "normal":                  float(cell_probs[0]),
                    "excessive_power_reduction": float(cell_probs[1]),
                    "interference":            float(cell_probs[2]),
                },
                timestamp    = ts,
                repair_action = REPAIR_ACTIONS[fault_name],
                is_anomaly   = is_anomaly,
            ))

        n_anomalies = sum(1 for d in detections if d.is_anomaly)
        self._total_inferences += 1

        return InferenceResult(
            timestamp    = ts,
            n_cells      = self._n_cells,
            n_anomalies  = n_anomalies,
            detections   = detections,
            learned_adj  = adj_np,
        )

    def ingest_batch(
        self,
        kpi_stream: np.ndarray,  # (T, n_cells, n_kpis)
        verbose:    bool = True,
    ) -> List[InferenceResult]:
        """
        Process a full KPI stream in batch mode.
        Useful for offline evaluation and demo.

        Returns list of InferenceResult (one per inference trigger).
        """
        results = []
        T = kpi_stream.shape[0]
        for t in range(T):
            result = self.ingest(kpi_stream[t])
            if result is not None:
                results.append(result)
                if verbose and result.n_anomalies > 0:
                    print(result.summary())
                    for d in result.anomalous_cells:
                        print(f"  {d}")
        return results

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_timesteps_ingested": self._buffer.n_ingested,
            "total_inferences_run":     self._total_inferences,
            "buffer_fill":              len(self._buffer.buffer),
            "window_size":              self._window_size,
            "device":                   str(self._device),
        }
