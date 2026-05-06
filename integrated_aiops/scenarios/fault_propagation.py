"""
Cross-Domain Fault Propagation Engine
======================================
Simulates realistic fault scenarios that cascade across optical → IP → RAN.

For each scenario it produces two outputs simultaneously:
  1. A list of TMF642-compatible alarm events (for alarm normalizer pipeline)
  2. KPI degradation profiles per cell per timestep (for SIMBA pipeline)

This is the bridge between the two projects — the same physical event
drives both the alarm stream and the KPI degradation.

Scenarios implemented:
  A. Fiber cut (Mumbai-Chennai span) — cascades to IP link loss → RAN cell OOS
  B. RAN equipment fault (RRH failure on single gNB)
  C. IP router CPU high (performance degradation, not outage)
  D. Multi-domain compound fault (fiber + compute)
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topology.unified_topology import (
    ALL_CELLS, N_CELLS, GNB_LIST, GNB_BY_ID, CELL_BY_INDEX,
    OPTICAL_NODES, IP_NODES, FIBER_SPANS, SPAN_TO_PE, PE_TO_CELLS,
    cells_affected_by_span, GNB_BACKHAUL,
    KPI_NAMES, N_KPIS
)

# ─────────────────────────────────────────────────────────────────────────────
# KPI baseline (normal operating values per KPI)
# ─────────────────────────────────────────────────────────────────────────────

KPI_BASELINE = {
    "rsrp_dbm":           {"mean": -80.0, "std": 6.0,  "min": -110.0, "max": -44.0},
    "rsrq_db":            {"mean": -10.0, "std": 2.5,  "min":  -20.0, "max":  -3.0},
    "sinr_db":            {"mean":  15.0, "std": 4.0,  "min":   -5.0, "max":  35.0},
    "dl_throughput_mbps": {"mean":  80.0, "std": 15.0, "min":    0.1, "max": 300.0},
    "ul_throughput_mbps": {"mean":  20.0, "std": 6.0,  "min":    0.1, "max": 100.0},
    "dl_bler_pct":        {"mean":   2.0, "std": 0.8,  "min":    0.0, "max":  10.0},
    "ul_bler_pct":        {"mean":   2.0, "std": 0.8,  "min":    0.0, "max":  10.0},
    "connected_ues":      {"mean":  15.0, "std": 4.0,  "min":    0.0, "max":  50.0},
    "handover_rate":      {"mean":   0.5, "std": 0.15, "min":    0.0, "max":   5.0},
}


# ─────────────────────────────────────────────────────────────────────────────
# Alarm event structure (TMF642-compatible)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlarmEvent:
    """A single normalised alarm event compatible with the TMF642 pipeline."""
    alarm_id:         str
    domain:           str         # optical / ip / ran / compute
    vendor:           str
    source_system:    str
    device_id:        str
    device_name:      str
    alarm_type:       str         # ITU-T X.733 alarmType
    perceived_severity: str       # critical / major / minor / warning / cleared
    probable_cause:   str         # ITU-T X.733 probableCause
    specific_problem: str
    alarm_details:    str
    service_affecting: bool
    is_root_cause:    bool
    raised_time:      str
    cleared_time:     Optional[str] = None
    state:            str = "raised"
    raw_format:       str = "json_rest"
    scenario_id:      str = ""
    propagated_from:  Optional[str] = None  # alarm_id of parent alarm


# ─────────────────────────────────────────────────────────────────────────────
# KPI degradation profiles
# ─────────────────────────────────────────────────────────────────────────────

def apply_backhaul_loss(kpis: np.ndarray, severity: float) -> np.ndarray:
    """
    KPI degradation when a cell loses its backhaul (transport failure).
    severity 0..1 scales the effect.
    Characteristic pattern: throughput collapses, BLER spikes, UEs drop.
    """
    k = kpis.copy()
    s = severity
    k[KPI_NAMES.index("rsrp_dbm")]           -= s * np.random.uniform(5, 10)
    k[KPI_NAMES.index("rsrq_db")]            -= s * np.random.uniform(2, 5)
    k[KPI_NAMES.index("sinr_db")]            -= s * np.random.uniform(4, 10)
    k[KPI_NAMES.index("dl_throughput_mbps")] *= max(0.05, 1 - s * 0.95)
    k[KPI_NAMES.index("ul_throughput_mbps")] *= max(0.05, 1 - s * 0.95)
    k[KPI_NAMES.index("dl_bler_pct")]        += s * np.random.uniform(20, 45)
    k[KPI_NAMES.index("ul_bler_pct")]        += s * np.random.uniform(15, 35)
    k[KPI_NAMES.index("connected_ues")]      *= max(0.1, 1 - s * 0.85)
    k[KPI_NAMES.index("handover_rate")]      += s * np.random.uniform(0.5, 2.0)
    return k

def apply_rrh_fault(kpis: np.ndarray, severity: float) -> np.ndarray:
    """
    KPI degradation from RRH hardware fault (power reduction type).
    Characteristic: RSRP/SINR drop, throughput moderate reduction, BLER moderate increase.
    """
    k = kpis.copy()
    s = severity
    k[KPI_NAMES.index("rsrp_dbm")]           -= s * np.random.uniform(15, 25)
    k[KPI_NAMES.index("rsrq_db")]            -= s * np.random.uniform(3, 6)
    k[KPI_NAMES.index("sinr_db")]            -= s * np.random.uniform(8, 15)
    k[KPI_NAMES.index("dl_throughput_mbps")] *= max(0.15, 1 - s * 0.70)
    k[KPI_NAMES.index("ul_throughput_mbps")] *= max(0.15, 1 - s * 0.60)
    k[KPI_NAMES.index("dl_bler_pct")]        += s * np.random.uniform(8, 20)
    k[KPI_NAMES.index("ul_bler_pct")]        += s * np.random.uniform(6, 15)
    k[KPI_NAMES.index("connected_ues")]      *= max(0.3, 1 - s * 0.50)
    return k

def apply_interference(kpis: np.ndarray, severity: float) -> np.ndarray:
    """
    KPI degradation from external interference.
    Characteristic: SINR collapses severely, BLER spikes, throughput crashes.
    """
    k = kpis.copy()
    s = severity
    k[KPI_NAMES.index("sinr_db")]            -= s * np.random.uniform(15, 30)
    k[KPI_NAMES.index("rsrq_db")]            -= s * np.random.uniform(4, 8)
    k[KPI_NAMES.index("dl_throughput_mbps")] *= max(0.05, 1 - s * 0.85)
    k[KPI_NAMES.index("ul_throughput_mbps")] *= max(0.10, 1 - s * 0.75)
    k[KPI_NAMES.index("dl_bler_pct")]        += s * np.random.uniform(15, 40)
    k[KPI_NAMES.index("ul_bler_pct")]        += s * np.random.uniform(10, 30)
    k[KPI_NAMES.index("handover_rate")]      += s * np.random.uniform(1.0, 3.0)
    return k

def clip_kpis(kpis: np.ndarray) -> np.ndarray:
    for i, name in enumerate(KPI_NAMES):
        b = KPI_BASELINE[name]
        kpis[i] = np.clip(kpis[i], b["min"], b["max"])
    return kpis


# ─────────────────────────────────────────────────────────────────────────────
# AR(1) baseline generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_baseline_kpis(n_timesteps: int, seed: int = 42) -> np.ndarray:
    """
    Generate normal KPI baseline for all 21 cells.
    Returns shape (n_timesteps, N_CELLS, N_KPIS).
    """
    rng = np.random.RandomState(seed)
    data = np.zeros((n_timesteps, N_CELLS, N_KPIS), dtype=np.float32)
    ar_coef = 0.85

    for c in range(N_CELLS):
        for k, name in enumerate(KPI_NAMES):
            cfg = KPI_BASELINE[name]
            # Small per-cell offset for diversity
            cell_mean = cfg["mean"] + rng.normal(0, cfg["std"] * 0.15)
            noise_std = cfg["std"] * np.sqrt(1 - ar_coef**2)
            series = np.zeros(n_timesteps)
            series[0] = cell_mean + rng.normal(0, cfg["std"])
            for t in range(1, n_timesteps):
                series[t] = (cell_mean
                             + ar_coef * (series[t-1] - cell_mean)
                             + rng.normal(0, noise_std))
            series = np.clip(series, cfg["min"], cfg["max"])
            data[:, c, k] = series

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Fault scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

def ts_str(base_dt: datetime, offset_s: int) -> str:
    return (base_dt + timedelta(seconds=offset_s)).strftime("%Y-%m-%dT%H:%M:%SZ")

@dataclass
class FaultScenario:
    """
    A complete cross-domain fault scenario.
    Produces both alarm events and KPI degradation profiles.
    """
    scenario_id:    str
    name:           str
    description:    str
    duration_s:     int
    fault_start_s:  int   # when the fault begins
    fault_end_s:    int   # when the fault clears (0 = does not clear)
    alarms:         List[AlarmEvent] = field(default_factory=list)

    # KPI fault injection schedule — list of (cell_index, fault_fn, start_s, end_s, severity)
    kpi_faults: List[Tuple] = field(default_factory=list)

    # SIMBA labels — list of (cell_index, label, start_s, end_s)
    # label: 0=normal, 1=power_reduction(backhaul/rrh), 2=interference
    labels: List[Tuple] = field(default_factory=list)


def build_fiber_cut_scenario(base_dt: datetime) -> FaultScenario:
    """
    Scenario A — Fiber cut on Mumbai-Chennai span.

    Timeline:
      t=0      Normal operation
      t=60s    Optical OSNR degradation warning (Nokia 1830PSS)
      t=75s    LOS alarm fires on ROADM-MUM-01 (Nokia 1830PSS)
      t=75s    Amplifier fault on AMP-MUM-CHN-01 (Nokia 1830PSS)
      t=85s    IP link down on RTR-PE-MUM-01 (Cisco syslog)
      t=88s    BGP session down on RTR-PE-MUM-01 (Cisco syslog)
      t=90s    Cells on gNB-MUM-SITE-A01 go OOS (Nokia NetAct)
      t=90s    Cells on gNB-MUM-SITE-A02 go OOS (Nokia NetAct)
      t=92s    CPRI failure on gNB-MUM-SITE-B01 (Ericsson ENM)
      t=600s   Fiber repaired — all alarms clear
    """
    scenario_id = "SCENARIO-A-FIBER-CUT"
    duration    = 900
    fault_start = 60
    fault_clear = 600

    alarms = [
        # ── Optical layer ────────────────────────────────────────────────
        AlarmEvent(
            alarm_id="ALM-A-OPT-001", domain="optical", vendor="Nokia",
            source_system="Nokia-1830PSS", device_id="ROADM-MUM-01",
            device_name="ROADM Mumbai 01",
            alarm_type="qualityOfServiceAlarm",
            perceived_severity="major",
            probable_cause="signalQualityEvaluationFailure",
            specific_problem="OSNR_DEGRADATION",
            alarm_details="OSNR on OCH-1-1-1-RX degrading. Value: 14.2 dB (threshold: 18.0 dB)",
            service_affecting=True, is_root_cause=False,
            raised_time=ts_str(base_dt, fault_start),
            cleared_time=ts_str(base_dt, fault_clear),
            scenario_id=scenario_id,
        ),
        AlarmEvent(
            alarm_id="ALM-A-OPT-002", domain="optical", vendor="Nokia",
            source_system="Nokia-1830PSS", device_id="ROADM-MUM-01",
            device_name="ROADM Mumbai 01",
            alarm_type="communicationsAlarm",
            perceived_severity="critical",
            probable_cause="lossOfSignal",
            specific_problem="LOS",
            alarm_details="Loss of Signal on OCH-1-1-1-TX. Rx Power: -40.0 dBm. Fiber span MUM-CHN-SPAN-1.",
            service_affecting=True, is_root_cause=True,
            raised_time=ts_str(base_dt, fault_start + 15),
            cleared_time=ts_str(base_dt, fault_clear),
            scenario_id=scenario_id,
        ),
        AlarmEvent(
            alarm_id="ALM-A-OPT-003", domain="optical", vendor="Nokia",
            source_system="Nokia-1830PSS", device_id="AMP-MUM-CHN-01",
            device_name="EDFA Amplifier Mum-Chn Span1",
            alarm_type="equipmentAlarm",
            perceived_severity="critical",
            probable_cause="equipmentFailure",
            specific_problem="AMPLIFIER_FAULT",
            alarm_details="EDFA output power below threshold: -35 dBm. Expected: +3 dBm.",
            service_affecting=True, is_root_cause=False,
            raised_time=ts_str(base_dt, fault_start + 15),
            cleared_time=ts_str(base_dt, fault_clear),
            scenario_id=scenario_id,
            propagated_from="ALM-A-OPT-002",
        ),
        # ── IP layer ─────────────────────────────────────────────────────
        AlarmEvent(
            alarm_id="ALM-A-IP-001", domain="ip", vendor="Cisco",
            source_system="Cisco-EPN-Manager", device_id="RTR-PE-MUM-01",
            device_name="PE Router Mumbai 01",
            alarm_type="communicationsAlarm",
            perceived_severity="major",
            probable_cause="communicationsSubsystemFailure",
            specific_problem="LINK-3-UPDOWN",
            alarm_details="Interface TenGigE0/0/0/1 changed state to down. Underlaid optical circuit lost.",
            service_affecting=True, is_root_cause=False,
            raised_time=ts_str(base_dt, fault_start + 25),
            cleared_time=ts_str(base_dt, fault_clear),
            scenario_id=scenario_id,
            propagated_from="ALM-A-OPT-002",
        ),
        AlarmEvent(
            alarm_id="ALM-A-IP-002", domain="ip", vendor="Cisco",
            source_system="Cisco-EPN-Manager", device_id="RTR-PE-MUM-01",
            device_name="PE Router Mumbai 01",
            alarm_type="communicationsAlarm",
            perceived_severity="major",
            probable_cause="softwareProgramAbnormallyTerminated",
            specific_problem="BGP-5-ADJCHG",
            alarm_details="BGP neighbour 10.1.2.1 (RTR-PE-CHN-01) down. Reason: hold-timer expired.",
            service_affecting=True, is_root_cause=False,
            raised_time=ts_str(base_dt, fault_start + 28),
            cleared_time=ts_str(base_dt, fault_clear),
            scenario_id=scenario_id,
            propagated_from="ALM-A-IP-001",
        ),
        # ── RAN layer ────────────────────────────────────────────────────
        AlarmEvent(
            alarm_id="ALM-A-RAN-001", domain="ran", vendor="Nokia",
            source_system="Nokia-NetAct", device_id="gNB-MUM-SITE-A01",
            device_name="gNB Mumbai Alpha 01",
            alarm_type="communicationsAlarm",
            perceived_severity="critical",
            probable_cause="communicationsSubsystemFailure",
            specific_problem="CELL_OUTAGE",
            alarm_details="NR cells on gNB-MUM-SITE-A01 out of service. Backhaul transport failure on N2/N3 path.",
            service_affecting=True, is_root_cause=False,
            raised_time=ts_str(base_dt, fault_start + 30),
            cleared_time=ts_str(base_dt, fault_clear),
            scenario_id=scenario_id,
            propagated_from="ALM-A-IP-001",
        ),
        AlarmEvent(
            alarm_id="ALM-A-RAN-002", domain="ran", vendor="Nokia",
            source_system="Nokia-NetAct", device_id="gNB-MUM-SITE-A02",
            device_name="gNB Mumbai Alpha 02",
            alarm_type="communicationsAlarm",
            perceived_severity="critical",
            probable_cause="communicationsSubsystemFailure",
            specific_problem="CELL_OUTAGE",
            alarm_details="NR cells on gNB-MUM-SITE-A02 out of service. Backhaul transport failure.",
            service_affecting=True, is_root_cause=False,
            raised_time=ts_str(base_dt, fault_start + 30),
            cleared_time=ts_str(base_dt, fault_clear),
            scenario_id=scenario_id,
            propagated_from="ALM-A-IP-001",
        ),
        AlarmEvent(
            alarm_id="ALM-A-RAN-003", domain="ran", vendor="Ericsson",
            source_system="Ericsson-ENM", device_id="gNB-MUM-SITE-B01",
            device_name="gNB Mumbai Beta 01",
            alarm_type="communicationsAlarm",
            perceived_severity="critical",
            probable_cause="transmissionError",
            specific_problem="CPRI_FAILURE",
            alarm_details="CPRI interface failure on gNB-MUM-SITE-B01. Transport loss detected.",
            service_affecting=True, is_root_cause=False,
            raised_time=ts_str(base_dt, fault_start + 32),
            cleared_time=ts_str(base_dt, fault_clear),
            scenario_id=scenario_id,
            propagated_from="ALM-A-IP-001",
        ),
    ]

    # KPI degradation — cells affected by MUM-CHN-SPAN-1 cut
    affected_cells = cells_affected_by_span("MUM-CHN-SPAN-1")
    ramp_start = fault_start + 25  # KPI degrades slightly after optical alarm
    ramp_end   = fault_start + 35  # full degradation within 10s
    kpi_faults = [
        (ci, apply_backhaul_loss, ramp_start, fault_clear, 0.90)
        for ci in affected_cells
    ]

    # SIMBA labels — backhaul loss maps to label 1 (power_reduction type)
    # because the KPI signature resembles excessive power reduction
    labels = [
        (ci, 1, ramp_start, fault_clear)
        for ci in affected_cells
    ]

    return FaultScenario(
        scenario_id=scenario_id,
        name="Fiber Cut — Mumbai-Chennai Span",
        description="Complete fiber cut on MUM-CHN-SPAN-1 cascading through IP to RAN",
        duration_s=duration,
        fault_start_s=fault_start,
        fault_end_s=fault_clear,
        alarms=alarms,
        kpi_faults=kpi_faults,
        labels=labels,
    )


def build_rrh_fault_scenario(base_dt: datetime) -> FaultScenario:
    """
    Scenario B — RRH hardware fault on gNB-CHN-SITE-A01.
    Single-domain RAN fault, no propagation to IP or optical.
    """
    scenario_id = "SCENARIO-B-RRH-FAULT"
    fault_start = 120
    fault_clear = 400

    # Cells on gNB-CHN-SITE-A01 are indices 9,10,11
    affected_cells = [c.cell_index for c in GNB_BY_ID["gNB-CHN-SITE-A01"].cells]

    alarms = [
        AlarmEvent(
            alarm_id="ALM-B-RAN-001", domain="ran", vendor="Nokia",
            source_system="Nokia-NetAct", device_id="gNB-CHN-SITE-A01",
            device_name="gNB Chennai Alpha 01",
            alarm_type="equipmentAlarm",
            perceived_severity="major",
            probable_cause="equipmentFailure",
            specific_problem="RRH_FAULT",
            alarm_details="RRH unit temperature exceeded 85°C on gNB-CHN-SITE-A01 Sector 2. Hardware fault detected.",
            service_affecting=True, is_root_cause=True,
            raised_time=ts_str(base_dt, fault_start),
            cleared_time=ts_str(base_dt, fault_clear),
            scenario_id=scenario_id,
        ),
    ]

    kpi_faults = [
        (ci, apply_rrh_fault, fault_start, fault_clear, 0.80)
        for ci in affected_cells
    ]
    labels = [(ci, 1, fault_start, fault_clear) for ci in affected_cells]

    return FaultScenario(
        scenario_id=scenario_id,
        name="RRH Hardware Fault — Chennai gNB",
        description="RRH overtemperature on gNB-CHN-SITE-A01 causing power reduction",
        duration_s=600,
        fault_start_s=fault_start,
        fault_end_s=fault_clear,
        alarms=alarms,
        kpi_faults=kpi_faults,
        labels=labels,
    )


def build_interference_scenario(base_dt: datetime) -> FaultScenario:
    """
    Scenario C — External interference on Bangalore cells.
    """
    scenario_id = "SCENARIO-C-INTERFERENCE"
    fault_start = 200
    fault_clear = 500

    affected_cells = [c.cell_index for c in GNB_BY_ID["gNB-BLR-SITE-A01"].cells]

    alarms = [
        AlarmEvent(
            alarm_id="ALM-C-RAN-001", domain="ran", vendor="Nokia",
            source_system="Nokia-NetAct", device_id="gNB-BLR-SITE-A01",
            device_name="gNB Bangalore Alpha 01",
            alarm_type="qualityOfServiceAlarm",
            perceived_severity="major",
            probable_cause="signalQualityEvaluationFailure",
            specific_problem="INTERFERENCE_DETECTED",
            alarm_details="SINR below threshold on gNB-BLR-SITE-A01. Possible external interference on n78 band.",
            service_affecting=True, is_root_cause=True,
            raised_time=ts_str(base_dt, fault_start),
            cleared_time=ts_str(base_dt, fault_clear),
            scenario_id=scenario_id,
        ),
    ]

    kpi_faults = [
        (ci, apply_interference, fault_start, fault_clear, 0.85)
        for ci in affected_cells
    ]
    labels = [(ci, 2, fault_start, fault_clear) for ci in affected_cells]

    return FaultScenario(
        scenario_id=scenario_id,
        name="External Interference — Bangalore",
        description="External interference source degrading Bangalore gNB cells",
        duration_s=700,
        fault_start_s=fault_start,
        fault_end_s=fault_clear,
        alarms=alarms,
        kpi_faults=kpi_faults,
        labels=labels,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Combined dataset generator
# ─────────────────────────────────────────────────────────────────────────────

class IntegratedDatasetGenerator:
    """
    Generates a combined dataset from multiple fault scenarios.
    Produces both KPI stream (for SIMBA) and alarm events (for normalizer).
    """

    def __init__(self, duration_s: int = 3600, seed: int = 42):
        self.duration_s = duration_s
        self.seed       = seed
        np.random.seed(seed)

    def generate(self) -> Dict:
        """
        Run all three scenarios sequentially within the duration window.
        Returns dict with:
          kpi_data      : (duration, N_CELLS, N_KPIS) float32
          labels        : (duration, N_CELLS) int64
          alarm_events  : List[AlarmEvent]
          scenarios     : List[FaultScenario]
        """
        base_dt = datetime.utcnow()

        # Build baseline KPI stream for full duration
        kpi_data = generate_baseline_kpis(self.duration_s, self.seed)
        labels   = np.zeros((self.duration_s, N_CELLS), dtype=np.int64)

        # Build scenarios with non-overlapping time offsets
        scenarios = [
            build_fiber_cut_scenario(base_dt),
            build_rrh_fault_scenario(
                base_dt + timedelta(seconds=self.duration_s // 3)
            ),
            build_interference_scenario(
                base_dt + timedelta(seconds=2 * self.duration_s // 3)
            ),
        ]

        # Time offsets so scenarios are spread across the duration
        offsets = [0, self.duration_s // 3, 2 * self.duration_s // 3]

        all_alarms = []
        for scenario, offset in zip(scenarios, offsets):
            # Apply KPI faults
            for cell_idx, fault_fn, start_s, end_s, severity in scenario.kpi_faults:
                abs_start = offset + start_s
                abs_end   = min(offset + end_s, self.duration_s)
                ramp_len  = 5
                for t in range(abs_start, abs_end):
                    if t >= self.duration_s:
                        break
                    ramp = min(1.0, (t - abs_start + 1) / ramp_len) * severity
                    kpi_data[t, cell_idx, :] = clip_kpis(
                        fault_fn(kpi_data[t, cell_idx, :], ramp)
                    )

            # Apply labels
            for cell_idx, label, start_s, end_s in scenario.labels:
                abs_start = offset + start_s
                abs_end   = min(offset + end_s, self.duration_s)
                labels[abs_start:abs_end, cell_idx] = label

            all_alarms.extend(scenario.alarms)

        print(f"Generated {self.duration_s}s KPI stream for {N_CELLS} cells")
        print(f"  Scenarios: {[s.name for s in scenarios]}")
        print(f"  Alarm events: {len(all_alarms)}")
        print(f"  Anomaly rate: {(labels > 0).mean():.3%}")

        return {
            "kpi_data":     kpi_data,
            "labels":       labels,
            "alarm_events": all_alarms,
            "scenarios":    scenarios,
            "offsets":      offsets,
            "base_dt":      base_dt,
            "duration_s":   self.duration_s,
        }
