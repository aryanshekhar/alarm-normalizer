"""
MonitorAgent
============
Background polling loop that calls run_inference every N seconds and
fires an on_anomaly(alert) callback when confidence exceeds threshold.
Deduplicates alerts per cell_id within a 5-minute window.
"""
import logging
import threading
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import model_store

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.75
DEDUP_WINDOW_S       = 300   # 5 minutes


@dataclass
class Alert:
    cell_id:    str
    gnb_id:     str
    confidence: float
    severity:   str
    fault_type: str
    kpi_values: dict
    timestamp:  str

    def to_dict(self) -> dict:
        return asdict(self)


class MonitorAgent:
    """
    Polls SIMBA inference on a fixed interval.
    Fires on_anomaly(Alert) for any high-confidence anomaly not seen
    within the dedup window.
    """

    def __init__(
        self,
        poll_interval_s: int = 10,
        on_anomaly: Optional[Callable[[Alert], None]] = None,
    ) -> None:
        self._poll_interval   = poll_interval_s
        self._on_anomaly      = on_anomaly
        self._running         = False
        self._thread: Optional[threading.Thread] = None

        # cell_id → last alert unix timestamp (for dedup)
        self._seen:   dict[str, float] = {}
        # cell_id → Alert (most recent firing per cell)
        self._active: dict[str, Alert] = {}
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def register_callback(self, callback: Callable[[Alert], None]) -> None:
        self._on_anomaly = callback

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="MonitorAgent"
        )
        self._thread.start()
        logger.info("MonitorAgent started (poll_interval=%ds)", self._poll_interval)

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._poll_interval + 5)
        logger.info("MonitorAgent stopped")

    @property
    def current_anomalies(self) -> list[Alert]:
        """Alerts fired within the last dedup window."""
        cutoff = time.time() - DEDUP_WINDOW_S
        with self._lock:
            return [
                alert for cell_id, alert in self._active.items()
                if self._seen.get(cell_id, 0) >= cutoff
            ]

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            try:
                self._poll()
            except Exception:
                logger.exception("MonitorAgent: poll error")
            time.sleep(self._poll_interval)

    def _poll(self) -> None:
        if not model_store.is_ready():
            return

        # Late import — keeps module loadable before torch is available
        from mcp.tools import run_inference, RunInferenceRequest
        result = run_inference(RunInferenceRequest(kpi_window="anomalous"))

        now = time.time()
        for anomaly in result.get("anomalies", []):
            if anomaly["confidence"] < CONFIDENCE_THRESHOLD:
                continue

            cell_id = anomaly["cell_id"]

            with self._lock:
                last_seen = self._seen.get(cell_id, 0)
                if now - last_seen < DEDUP_WINDOW_S:
                    continue  # duplicate within window

            alert = Alert(
                cell_id    = cell_id,
                gnb_id     = anomaly["gnb_id"],
                confidence = anomaly["confidence"],
                severity   = anomaly["severity"],
                fault_type = anomaly["fault_type"],
                kpi_values = anomaly.get("kpi_values", {}),
                timestamp  = anomaly["timestamp"],
            )

            with self._lock:
                self._seen[cell_id]   = now
                self._active[cell_id] = alert

            logger.info(
                "ALERT cell=%s gnb=%s fault=%s conf=%.3f sev=%s",
                alert.cell_id, alert.gnb_id,
                alert.fault_type, alert.confidence, alert.severity,
            )

            if self._on_anomaly:
                try:
                    self._on_anomaly(alert)
                except Exception:
                    logger.exception("MonitorAgent: on_anomaly callback raised")
