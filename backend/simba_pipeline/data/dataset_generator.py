"""
Synthetic 5G RAN KPI Dataset Generator
=======================================
Generates KPI time series data structurally identical to the Simu5G
output used in the SIMBA paper (arXiv:2406.15638).

Topology: 7-cell hexagonal eMBB-Urban deployment (3GPP TR 38.901)
  - 7 gNBs arranged in hexagonal grid, 200m inter-site distance
  - Each gNB has 3 sectors (21 cells total)
  - UEs distributed uniformly, mobile and static patterns

KPIs per cell (aggregated to 1-second intervals):
  - rsrp_dbm        : Reference Signal Received Power (dBm)
  - rsrq_db         : Reference Signal Received Quality (dB)
  - sinr_db         : Signal-to-Interference-plus-Noise Ratio (dB)
  - dl_throughput_mbps : Downlink throughput (Mbps)
  - ul_throughput_mbps : Uplink throughput (Mbps)
  - dl_bler_pct     : Downlink Block Error Rate (%)
  - ul_bler_pct     : Uplink Block Error Rate (%)
  - connected_ues   : Number of connected UEs
  - handover_rate   : Handovers per second

Fault types (matching SIMBA paper):
  - excessive_power_reduction : TX power dropped, RSRP/SINR degrade
  - interference              : External interference, SINR drops severely
  - normal                    : No fault

Usage:
    python data/dataset_generator.py --cells 7 --duration 3600 --output data/kpi_dataset.npz
"""

import numpy as np
import pandas as pd
import argparse
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

# Reproducibility
SEED = 42
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 3GPP eMBB-Urban parameters (calibrated per SIMBA paper / TR 38.901)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CellConfig:
    """Per-cell configuration parameters."""
    cell_id:       int
    gnb_id:        int
    sector:        int           # 1, 2, or 3
    x_pos:         float         # metres
    y_pos:         float         # metres
    tx_power_dbm:  float = 46.0  # 3GPP eMBB macro default
    freq_band:     str   = "n78"
    bandwidth_mhz: float = 100.0


def build_hexagonal_topology(n_sites: int = 7, isd_m: float = 200.0) -> List[CellConfig]:
    """
    Build a 7-site hexagonal topology with 3 sectors per site.
    Site 0 is at centre; sites 1-6 are arranged around it at distance isd_m.
    """
    # Hexagonal grid positions
    site_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        site_positions.append((isd_m * np.cos(angle), isd_m * np.sin(angle)))

    cells = []
    cell_id = 0
    for gnb_id, (x, y) in enumerate(site_positions[:n_sites]):
        for sector in range(1, 4):
            cells.append(CellConfig(
                cell_id=cell_id, gnb_id=gnb_id, sector=sector,
                x_pos=x, y_pos=y
            ))
            cell_id += 1
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# KPI baseline distributions (calibrated to 3GPP eMBB-Urban)
# ─────────────────────────────────────────────────────────────────────────────

# Normal operating range — mean, std per KPI
KPI_NORMAL = {
    "rsrp_dbm":           {"mean": -80.0,  "std":  8.0,  "min": -110.0, "max": -44.0},
    "rsrq_db":            {"mean": -10.0,  "std":  3.0,  "min":  -20.0, "max":  -3.0},
    "sinr_db":            {"mean":  15.0,  "std":  5.0,  "min":   -5.0, "max":  35.0},
    "dl_throughput_mbps": {"mean":  80.0,  "std": 20.0,  "min":    0.1, "max": 300.0},
    "ul_throughput_mbps": {"mean":  20.0,  "std":  8.0,  "min":    0.1, "max": 100.0},
    "dl_bler_pct":        {"mean":   2.0,  "std":  1.0,  "min":    0.0, "max":  10.0},
    "ul_bler_pct":        {"mean":   2.0,  "std":  1.0,  "min":    0.0, "max":  10.0},
    "connected_ues":      {"mean":  15.0,  "std":  5.0,  "min":    0.0, "max":  50.0},
    "handover_rate":      {"mean":   0.5,  "std":  0.2,  "min":    0.0, "max":   5.0},
}

KPI_NAMES = list(KPI_NORMAL.keys())
N_KPIS    = len(KPI_NAMES)  # 9


# ─────────────────────────────────────────────────────────────────────────────
# Fault effect profiles
# ─────────────────────────────────────────────────────────────────────────────

def apply_fault_effects(
    kpi_values: np.ndarray,  # shape (N_KPIS,)
    fault_type: str,
    fault_severity: float = 1.0,  # 0..1, scales the effect
) -> np.ndarray:
    """
    Apply fault-specific KPI degradation.  Returns modified KPI array.
    fault_severity=1.0 is full effect; 0.0 is no effect (used for gradual onset).
    """
    kpis = kpi_values.copy()
    s = fault_severity

    if fault_type == "excessive_power_reduction":
        # TX power reduced → RSRP drops, SINR drops, throughput drops, BLER rises
        kpis[KPI_NAMES.index("rsrp_dbm")]           -= s * np.random.uniform(15, 25)
        kpis[KPI_NAMES.index("rsrq_db")]             -= s * np.random.uniform(3, 6)
        kpis[KPI_NAMES.index("sinr_db")]             -= s * np.random.uniform(8, 15)
        kpis[KPI_NAMES.index("dl_throughput_mbps")]  *= max(0.1, 1 - s * 0.7)
        kpis[KPI_NAMES.index("ul_throughput_mbps")]  *= max(0.1, 1 - s * 0.6)
        kpis[KPI_NAMES.index("dl_bler_pct")]         += s * np.random.uniform(8, 20)
        kpis[KPI_NAMES.index("ul_bler_pct")]         += s * np.random.uniform(6, 15)
        kpis[KPI_NAMES.index("connected_ues")]       *= max(0.2, 1 - s * 0.5)

    elif fault_type == "interference":
        # External interference → SINR drops severely, BLER spikes
        kpis[KPI_NAMES.index("sinr_db")]             -= s * np.random.uniform(15, 30)
        kpis[KPI_NAMES.index("rsrq_db")]             -= s * np.random.uniform(4, 8)
        kpis[KPI_NAMES.index("dl_throughput_mbps")]  *= max(0.05, 1 - s * 0.85)
        kpis[KPI_NAMES.index("ul_throughput_mbps")]  *= max(0.1,  1 - s * 0.75)
        kpis[KPI_NAMES.index("dl_bler_pct")]         += s * np.random.uniform(15, 40)
        kpis[KPI_NAMES.index("ul_bler_pct")]         += s * np.random.uniform(10, 30)
        kpis[KPI_NAMES.index("handover_rate")]       += s * np.random.uniform(1, 3)

    return kpis


# ─────────────────────────────────────────────────────────────────────────────
# Time series generator
# ─────────────────────────────────────────────────────────────────────────────

class KPITimeSeriesGenerator:
    """
    Generates per-cell KPI time series with realistic temporal correlations
    and configurable fault injection.

    Each cell's KPIs follow an AR(1) process around the baseline to simulate
    realistic temporal autocorrelation seen in real 5G telemetry.
    """

    def __init__(self, cells: List[CellConfig], duration_s: int = 3600):
        self.cells      = cells
        self.n_cells    = len(cells)
        self.duration_s = duration_s
        self.n_timesteps = duration_s  # 1-second resolution

    def _generate_ar1_series(
        self,
        n_steps: int,
        mean: float,
        std: float,
        ar_coef: float = 0.85,  # temporal autocorrelation
    ) -> np.ndarray:
        """Generate AR(1) time series with given statistics."""
        noise = np.random.normal(0, std * np.sqrt(1 - ar_coef**2), n_steps)
        series = np.zeros(n_steps)
        series[0] = mean + np.random.normal(0, std)
        for t in range(1, n_steps):
            series[t] = mean + ar_coef * (series[t-1] - mean) + noise[t]
        return series

    def generate(
        self,
        fault_schedule: Optional[List[Dict]] = None,
        anomaly_fraction: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Generate full dataset.

        Args:
            fault_schedule: List of fault dicts with keys:
                cell_id, fault_type, start_t, end_t, severity
                If None, faults are injected randomly at anomaly_fraction rate.
            anomaly_fraction: Fraction of timesteps that are anomalous.

        Returns:
            kpi_data  : np.ndarray shape (n_timesteps, n_cells, n_kpis)
            labels    : np.ndarray shape (n_timesteps, n_cells) — 0=normal, 1=power, 2=interference
            fault_log : list of fault event dicts for ground truth reference
        """
        print(f"Generating {self.n_timesteps}s of KPI data for {self.n_cells} cells...")

        # Initialise with normal baseline
        kpi_data = np.zeros((self.n_timesteps, self.n_cells, N_KPIS))
        labels   = np.zeros((self.n_timesteps, self.n_cells), dtype=np.int64)

        # Generate baseline AR(1) series per cell per KPI
        for c in range(self.n_cells):
            for k, kpi_name in enumerate(KPI_NAMES):
                cfg = KPI_NORMAL[kpi_name]
                # Add small per-cell offset to create realistic diversity
                cell_mean = cfg["mean"] + np.random.normal(0, cfg["std"] * 0.2)
                series = self._generate_ar1_series(
                    self.n_timesteps, cell_mean, cfg["std"]
                )
                # Clip to physical bounds
                series = np.clip(series, cfg["min"], cfg["max"])
                kpi_data[:, c, k] = series

        # Build fault schedule if not provided
        if fault_schedule is None:
            fault_schedule = self._build_random_fault_schedule(anomaly_fraction)

        # Apply faults
        fault_type_to_label = {
            "normal": 0,
            "excessive_power_reduction": 1,
            "interference": 2,
        }
        fault_log = []
        for fault in fault_schedule:
            c         = fault["cell_id"]
            ft        = fault["fault_type"]
            start_t   = fault["start_t"]
            end_t     = fault["end_t"]
            severity  = fault.get("severity", 1.0)

            # Gradual onset (ramp up over 5 seconds)
            ramp_steps = min(5, end_t - start_t)
            for t in range(start_t, end_t):
                if t >= self.n_timesteps:
                    break
                ramp = min(1.0, (t - start_t + 1) / ramp_steps) * severity
                # Apply fault to each KPI independently with noise
                kpi_data[t, c, :] = apply_fault_effects(
                    kpi_data[t, c, :], ft, ramp
                )
                # Re-clip after fault application
                for k, kpi_name in enumerate(KPI_NAMES):
                    cfg = KPI_NORMAL[kpi_name]
                    kpi_data[t, c, k] = np.clip(
                        kpi_data[t, c, k], cfg["min"], cfg["max"]
                    )
                labels[t, c] = fault_type_to_label[ft]

            fault_log.append({
                "cell_id":    c,
                "fault_type": ft,
                "start_t":    start_t,
                "end_t":      end_t,
                "severity":   severity,
                "label":      fault_type_to_label[ft],
            })

        print(f"  Generated {len(fault_schedule)} fault events.")
        print(f"  Anomaly rate: {(labels > 0).mean():.3%}")
        return kpi_data, labels, fault_log

    def _build_random_fault_schedule(
        self, target_fraction: float
    ) -> List[Dict]:
        """Randomly inject faults to reach target_fraction anomaly rate."""
        fault_types = ["excessive_power_reduction", "interference"]
        # Average fault duration: 30-120 seconds
        avg_duration  = 60
        n_faults_needed = int(
            (target_fraction * self.n_timesteps * self.n_cells) / avg_duration
        )
        schedule = []
        for _ in range(max(1, n_faults_needed)):
            cell_id    = np.random.randint(0, self.n_cells)
            fault_type = np.random.choice(fault_types)
            start_t    = np.random.randint(0, self.n_timesteps - 120)
            duration   = np.random.randint(30, 120)
            end_t      = min(start_t + duration, self.n_timesteps)
            severity   = np.random.uniform(0.5, 1.0)
            schedule.append({
                "cell_id":    cell_id,
                "fault_type": fault_type,
                "start_t":    start_t,
                "end_t":      end_t,
                "severity":   severity,
            })
        return schedule


# ─────────────────────────────────────────────────────────────────────────────
# Adjacency matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def build_adjacency_matrix(
    cells: List[CellConfig],
    distance_threshold_m: float = 400.0,
) -> np.ndarray:
    """
    Build cell adjacency matrix based on physical proximity.
    Cells within distance_threshold_m of each other are connected.
    Also adds intra-site connections (same gNB, different sectors).

    Returns:
        adj : np.ndarray shape (n_cells, n_cells), binary, symmetric
    """
    n = len(cells)
    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ci, cj = cells[i], cells[j]
            # Intra-site (same gNB)
            if ci.gnb_id == cj.gnb_id:
                adj[i, j] = 1.0
                continue
            # Inter-site proximity
            dist = np.sqrt((ci.x_pos - cj.x_pos)**2 + (ci.y_pos - cj.y_pos)**2)
            if dist <= distance_threshold_m:
                adj[i, j] = 1.0
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Data split and windowing
# ─────────────────────────────────────────────────────────────────────────────

def create_sliding_windows(
    kpi_data: np.ndarray,  # (T, N_cells, N_kpis)
    labels:   np.ndarray,  # (T, N_cells)
    window_size: int = 30,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series input.

    Returns:
        X : np.ndarray (n_windows, window_size, n_cells, n_kpis)
        y : np.ndarray (n_windows, n_cells)  — label at window end step
    """
    T = kpi_data.shape[0]
    windows_X, windows_y = [], []
    for start in range(0, T - window_size, stride):
        end = start + window_size
        windows_X.append(kpi_data[start:end])
        windows_y.append(labels[end - 1])  # label is at the last timestep
    return np.array(windows_X, dtype=np.float32), np.array(windows_y, dtype=np.int64)


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.50,
    val_frac:   float = 0.25,
) -> Tuple:
    """
    Temporal split — preserves time ordering (no shuffling).
    50% train / 25% val / 25% test as per SIMBA paper.
    """
    n = len(X)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    return (
        X[:n_train],           y[:n_train],
        X[n_train:n_train+n_val], y[n_train:n_train+n_val],
        X[n_train+n_val:],     y[n_train+n_val:],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

class KPINormalizer:
    """Min-max normaliser per KPI dimension, fitted on training data."""

    def __init__(self):
        self.min_vals = None
        self.max_vals = None

    def fit(self, X_train: np.ndarray) -> "KPINormalizer":
        """X_train: (n, window, n_cells, n_kpis)"""
        flat = X_train.reshape(-1, X_train.shape[-1])  # (n*window*n_cells, n_kpis)
        self.min_vals = flat.min(axis=0)
        self.max_vals = flat.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        denom = self.max_vals - self.min_vals
        denom = np.where(denom == 0, 1.0, denom)
        return (X - self.min_vals) / denom

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def save(self, path: str) -> None:
        np.savez(path, min_vals=self.min_vals, max_vals=self.max_vals)

    @classmethod
    def load(cls, path: str) -> "KPINormalizer":
        data = np.load(path)
        n = cls()
        n.min_vals = data["min_vals"]
        n.max_vals = data["max_vals"]
        return n


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 5G KPI dataset")
    parser.add_argument("--n-sites",          type=int,   default=7,
                        help="Number of gNB sites (hexagonal grid)")
    parser.add_argument("--duration",         type=int,   default=3600,
                        help="Duration in seconds (default: 3600 = 1 hour)")
    parser.add_argument("--anomaly-fraction", type=float, default=0.02,
                        help="Fraction of timesteps to inject faults (default: 0.02)")
    parser.add_argument("--window-size",      type=int,   default=30,
                        help="Sliding window size in seconds (default: 30)")
    parser.add_argument("--output",           type=str,   default="data/kpi_dataset.npz",
                        help="Output .npz file path")
    args = parser.parse_args()

    # Build topology
    cells = build_hexagonal_topology(args.n_sites)
    adj   = build_adjacency_matrix(cells)
    print(f"Topology: {len(cells)} cells across {args.n_sites} gNB sites")
    print(f"Adjacency matrix: {adj.sum()} edges")

    # Generate raw time series
    gen      = KPITimeSeriesGenerator(cells, args.duration)
    kpi_data, labels, fault_log = gen.generate(
        anomaly_fraction=args.anomaly_fraction
    )
    print(f"Raw data shape: {kpi_data.shape}")  # (T, N_cells, N_kpis)

    # Create windows
    X, y = create_sliding_windows(kpi_data, labels, args.window_size)
    print(f"Windowed data shape: X={X.shape}, y={y.shape}")

    # Split
    X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(X, y)
    print(f"Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")

    # Normalise
    normalizer = KPINormalizer()
    X_tr  = normalizer.fit_transform(X_tr)
    X_val = normalizer.transform(X_val)
    X_te  = normalizer.transform(X_te)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        X_train=X_tr, y_train=y_tr,
        X_val=X_val,  y_val=y_val,
        X_test=X_te,  y_test=y_te,
        adjacency=adj,
        kpi_names=np.array(KPI_NAMES),
        n_cells=len(cells),
        n_kpis=N_KPIS,
    )
    normalizer.save(args.output.replace(".npz", "_normalizer.npz"))

    print(f"\nDataset saved to {args.output}")
    print(f"Class distribution (train):")
    for label_id, name in enumerate(["normal", "power_reduction", "interference"]):
        pct = (y_tr == label_id).mean() * 100
        print(f"  {name:25s}: {pct:.2f}%")


if __name__ == "__main__":
    main()
