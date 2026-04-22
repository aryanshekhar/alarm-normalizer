"""
Base Adapter Interface + shared lookup tables.

Every vendor adapter inherits from BaseAdapter and overrides parse().
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import uuid

from core.model import (
    AlarmType, PerceivedSeverity, NetworkDomain, CanonicalAlarm,
    AlarmedObjectRef, AlarmState
)


# ─────────────────────────────────────────────
# Severity mapping tables per source format
# ─────────────────────────────────────────────

# Syslog severity (0=emergency … 7=debug) → TMF642 PerceivedSeverity
SYSLOG_SEVERITY_MAP: Dict[int, PerceivedSeverity] = {
    0: PerceivedSeverity.CRITICAL,      # emergency
    1: PerceivedSeverity.CRITICAL,      # alert
    2: PerceivedSeverity.CRITICAL,      # critical
    3: PerceivedSeverity.MAJOR,         # error
    4: PerceivedSeverity.MINOR,         # warning
    5: PerceivedSeverity.WARNING,       # notice
    6: PerceivedSeverity.INDETERMINATE, # informational
    7: PerceivedSeverity.INDETERMINATE, # debug
}

# Generic string labels from vendor NMS APIs → TMF642 PerceivedSeverity
SEVERITY_STRING_MAP: Dict[str, PerceivedSeverity] = {
    # Standard TMF / ITU
    "critical": PerceivedSeverity.CRITICAL,
    "major": PerceivedSeverity.MAJOR,
    "minor": PerceivedSeverity.MINOR,
    "warning": PerceivedSeverity.WARNING,
    "indeterminate": PerceivedSeverity.INDETERMINATE,
    "cleared": PerceivedSeverity.CLEARED,
    # Cisco EPN Manager
    "critical": PerceivedSeverity.CRITICAL,
    "major": PerceivedSeverity.MAJOR,
    "minor": PerceivedSeverity.MINOR,
    "warning": PerceivedSeverity.WARNING,
    "info": PerceivedSeverity.INDETERMINATE,
    "normal": PerceivedSeverity.CLEARED,
    # Nokia NetAct
    "CRITICAL": PerceivedSeverity.CRITICAL,
    "MAJOR": PerceivedSeverity.MAJOR,
    "MINOR": PerceivedSeverity.MINOR,
    "WARNING": PerceivedSeverity.WARNING,
    "INDETERMINATE": PerceivedSeverity.INDETERMINATE,
    "CLEARED": PerceivedSeverity.CLEARED,
    # Ericsson FM
    "A1": PerceivedSeverity.CRITICAL,
    "A2": PerceivedSeverity.MAJOR,
    "A3": PerceivedSeverity.MINOR,
    "A4": PerceivedSeverity.WARNING,
    # Huawei iManager
    "Emergency": PerceivedSeverity.CRITICAL,
    "Alert": PerceivedSeverity.CRITICAL,
    "Critical": PerceivedSeverity.CRITICAL,
    "Major": PerceivedSeverity.MAJOR,
    "Minor": PerceivedSeverity.MINOR,
    "Warning": PerceivedSeverity.WARNING,
    "Cleared": PerceivedSeverity.CLEARED,
    # Numeric
    "1": PerceivedSeverity.CRITICAL,
    "2": PerceivedSeverity.MAJOR,
    "3": PerceivedSeverity.MINOR,
    "4": PerceivedSeverity.WARNING,
    "5": PerceivedSeverity.INDETERMINATE,
    "0": PerceivedSeverity.CLEARED,
}


def map_severity(raw: Any) -> PerceivedSeverity:
    """
    Convert any vendor severity representation to TMF642 PerceivedSeverity.
    Handles: int (syslog), str (various NMS labels), already-canonical values.
    """
    if isinstance(raw, PerceivedSeverity):
        return raw
    if isinstance(raw, int):
        return SYSLOG_SEVERITY_MAP.get(raw, PerceivedSeverity.INDETERMINATE)
    if isinstance(raw, str):
        return SEVERITY_STRING_MAP.get(raw.strip(), PerceivedSeverity.INDETERMINATE)
    return PerceivedSeverity.INDETERMINATE


# ─────────────────────────────────────────────
# ProbableCause normalisation
# ─────────────────────────────────────────────
# Maps vendor-specific probable cause strings → ITU-T X.733 / 3GPP values
# (representative subset — extend per deployment)

PROBABLE_CAUSE_MAP: Dict[str, str] = {
    # Cisco IOS syslog facility codes → probable cause
    "LINK-3-UPDOWN": "communicationsSubsystemFailure",
    "BGP-5-ADJCHANGE": "softwareProgramAbnormallyTerminated",
    "OSPF-5-ADJCHG": "softwareProgramAbnormallyTerminated",
    "SYS-5-CONFIG_I": "configurationOrCustomizationError",
    "LINEPROTO-5-UPDOWN": "communicationsSubsystemFailure",
    "HARDWARE-2-FAN_FAILURE": "coolingSystemFailure",
    "HARDWARE-3-CPU_OVERLOAD": "processorProblem",
    "IF-MIB::linkDown": "communicationsSubsystemFailure",
    "IF-MIB::linkUp": "communicationsSubsystemFailure",
    # Nokia NetAct FM probable causes
    "LINK_FAILURE": "communicationsSubsystemFailure",
    "HW_FAILURE": "equipmentFailure",
    "SW_ERROR": "softwareProgramError",
    "POWER_FAILURE": "powerProblem",
    "COOLING_FAILURE": "coolingSystemFailure",
    "CELL_OUTAGE": "communicationsSubsystemFailure",
    "RRH_FAULT": "equipmentFailure",
    "CPRI_FAILURE": "transmissionError",
    # Ericsson FM codes
    "CELL_DISABLED": "communicationsSubsystemFailure",
    "RADIO_UNIT_FAULT": "equipmentFailure",
    "BASEBAND_UNIT_FAULT": "equipmentFailure",
    "TRANSPORT_FAULT": "transmissionError",
    "CLOCK_SYNC_LOSS": "timingProblem",
    # Optical domain (Ciena, Infinera, Nokia OTN)
    "LOS": "lossOfSignal",
    "LOSS_OF_FRAME": "framingError",
    "SIGNAL_DEGRADE": "signalQualityEvaluationFailure",
    "AMPLIFIER_FAULT": "equipmentFailure",
    "SPAN_LOSS_CHANGE": "transmissionError",
    "FIBER_CUT": "lossOfSignal",
    "OSNR_DEGRADATION": "signalQualityEvaluationFailure",
    # IP domain
    "BGP_PEER_DOWN": "softwareProgramAbnormallyTerminated",
    "MPLS_LSP_DOWN": "communicationsSubsystemFailure",
    "INTERFACE_DOWN": "communicationsSubsystemFailure",
    "HIGH_CPU": "processorProblem",
    "HIGH_MEMORY": "storageCapacityProblem",
    "HIGH_BER": "transmissionError",
    # Compute / cloud
    "HOST_DOWN": "equipmentFailure",
    "VM_DOWN": "softwareProgramAbnormallyTerminated",
    "DISK_FULL": "storageCapacityProblem",
    "NIC_FAILURE": "communicationsSubsystemFailure",
    "POWER_REDUNDANCY_LOST": "powerProblem",
    # Threshold crossings
    "thresholdCrossed": "thresholdCrossed",
    "THRESHOLD_CROSSED": "thresholdCrossed",
    "Threshold crossed": "thresholdCrossed",
}


def map_probable_cause(raw: str) -> str:
    """Return canonical ITU-T X.733 probable cause string, or best-effort lower-cased raw."""
    if not raw:
        return "unspecified"
    canonical = PROBABLE_CAUSE_MAP.get(raw.strip())
    if canonical:
        return canonical
    # Try case-insensitive lookup
    raw_lower = raw.strip().lower()
    for k, v in PROBABLE_CAUSE_MAP.items():
        if k.lower() == raw_lower:
            return v
    # Return normalised raw value as fallback
    return raw.strip().replace(" ", "_").lower()


# ─────────────────────────────────────────────
# AlarmType inference heuristics
# ─────────────────────────────────────────────

def infer_alarm_type(
    probable_cause: str,
    alarmed_object_type: Optional[str] = None,
    specific_problem: Optional[str] = None,
) -> AlarmType:
    """
    Infer ITU-T X.733 alarm type from probable cause and object context.
    """
    pc = probable_cause.lower()
    if any(k in pc for k in ["communication", "link", "interface", "transmission",
                              "signal", "loss_of", "fiber", "transport", "bgp"]):
        return AlarmType.COMMUNICATIONS_ALARM
    if any(k in pc for k in ["equipment", "hardware", "fan", "power", "cooling",
                              "rru", "rrh", "amplifier"]):
        return AlarmType.EQUIPMENT_ALARM
    if any(k in pc for k in ["software", "program", "crash", "restart",
                              "vm_down", "process"]):
        return AlarmType.PROCESSING_ERROR_ALARM
    if any(k in pc for k in ["threshold", "qos", "utilization",
                              "congestion", "latency", "packet_loss"]):
        return AlarmType.QUALITY_OF_SERVICE_ALARM
    if any(k in pc for k in ["temperature", "humidity", "door", "smoke",
                              "flood", "environmental"]):
        return AlarmType.ENVIRONMENTAL_ALARM
    if any(k in pc for k in ["timing", "clock", "sync"]):
        return AlarmType.TIME_DOMAIN_VIOLATION
    if any(k in pc for k in ["security", "auth", "unauthorized", "intrusion"]):
        return AlarmType.SECURITY_SERVICE_VIOLATION
    return AlarmType.COMMUNICATIONS_ALARM  # safe default for network equipment


# ─────────────────────────────────────────────
# Base Adapter
# ─────────────────────────────────────────────

class BaseAdapter(ABC):
    """
    Contract that every vendor adapter must implement.
    Subclasses receive a raw parsed payload (dict/str)
    and return a CanonicalAlarm.
    """

    VENDOR: str = "generic"
    DOMAIN: NetworkDomain = NetworkDomain.UNKNOWN
    RAW_FORMAT: str = "unknown"

    def normalise(self, raw: Any) -> CanonicalAlarm:
        """
        Public entry point.  Calls parse(), then stamps provenance metadata.
        """
        alarm = self.parse(raw)
        alarm.x_vendor = self.VENDOR
        alarm.x_domain = self.DOMAIN
        alarm.x_raw_format = self.RAW_FORMAT
        alarm.x_raw_payload = raw if isinstance(raw, dict) else {"raw": str(raw)}
        if not alarm.alarm_reporting_time:
            alarm.alarm_reporting_time = datetime.now(timezone.utc).replace(tzinfo=None)
        return alarm

    @abstractmethod
    def parse(self, raw: Any) -> CanonicalAlarm:
        """
        Vendor-specific parsing logic.
        Must return a CanonicalAlarm with at minimum:
          id, alarm_raised_time, alarm_type, perceived_severity,
          alarmed_object, probable_cause, state
        """
        ...

    @staticmethod
    def _new_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _now() -> datetime:
        return datetime.utcnow()

    @staticmethod
    def _parse_ts(ts_str: str) -> datetime:
        """Parse ISO-8601 or common vendor timestamp strings."""
        if not ts_str:
            return datetime.utcnow()
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
        ):
            try:
                return datetime.strptime(ts_str.strip().rstrip("Z"), fmt.rstrip("Z"))
            except ValueError:
                continue
        return datetime.utcnow()
