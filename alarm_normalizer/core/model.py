"""
TMF642 Canonical Alarm Model
Based on TM Forum TMF642 Alarm Management API v4.0 / ITU-T X.733
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import uuid


# ─────────────────────────────────────────────
# ITU-T X.733 / TMF642 Enumerations
# ─────────────────────────────────────────────

class PerceivedSeverity(str, Enum):
    """ITU-T X.733 §8.1.2.3 — consistent with TMF642"""
    CRITICAL      = "critical"
    MAJOR         = "major"
    MINOR         = "minor"
    WARNING       = "warning"
    INDETERMINATE = "indeterminate"
    CLEARED       = "cleared"


class AlarmType(str, Enum):
    """ITU-T X.733 §8.1.1 — categorises the alarm"""
    COMMUNICATIONS_ALARM            = "communicationsAlarm"
    PROCESSING_ERROR_ALARM          = "processingErrorAlarm"
    ENVIRONMENTAL_ALARM             = "environmentalAlarm"
    QUALITY_OF_SERVICE_ALARM        = "qualityOfServiceAlarm"
    EQUIPMENT_ALARM                 = "equipmentAlarm"
    INTEGRITY_VIOLATION             = "integrityViolation"
    OPERATIONAL_VIOLATION           = "operationalViolation"
    PHYSICAL_VIOLATION              = "physicalViolation"
    SECURITY_SERVICE_VIOLATION      = "securityServiceOrMechanismViolation"
    TIME_DOMAIN_VIOLATION           = "timeDomainViolation"


class AckState(str, Enum):
    """TMF642 acknowledgement states"""
    ACKNOWLEDGED   = "acknowledged"
    UNACKNOWLEDGED = "unacknowledged"


class AlarmState(str, Enum):
    """TMF642 alarm lifecycle states"""
    RAISED    = "raised"
    UPDATED   = "updated"
    CLEARED   = "cleared"


class NetworkDomain(str, Enum):
    """Network domain — extension to TMF642 for multi-domain support"""
    RAN     = "ran"
    IP      = "ip"
    OPTICAL = "optical"
    COMPUTE = "compute"
    CORE    = "core"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────
# Supporting sub-objects (TMF642 §4)
# ─────────────────────────────────────────────

@dataclass
class AlarmedObjectRef:
    """Reference to the managed object that raised the alarm"""
    id: str                          # NE/device identifier
    href: Optional[str] = None       # REST reference URI
    name: Optional[str] = None       # human-readable name
    referred_type: Optional[str] = None  # MO class name (e.g. "gNB", "Router")


@dataclass
class ServiceRef:
    """Affected service reference"""
    id: str
    href: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AlarmRef:
    """Reference to a related/parent alarm"""
    id: str
    href: Optional[str] = None


@dataclass
class ThresholdCrossedInfo:
    """Populated for QoS alarms with threshold violations (TMF642 §4)"""
    threshold_indicator: str         # KPI name, e.g. "cpuUtilization"
    observed_value: float
    threshold_value: float
    direction: str                   # "up" | "down"
    granularity: Optional[str] = None  # "1min", "5min", "15min", "1hour"
    indicator_unit: Optional[str] = None


@dataclass
class Comment:
    """Operator comment on an alarm"""
    comment: str
    system_id: Optional[str] = None
    user_id: Optional[str] = None
    time: Optional[datetime] = None


# ─────────────────────────────────────────────
# Canonical Alarm — TMF642 compliant
# ─────────────────────────────────────────────

@dataclass
class CanonicalAlarm:
    """
    TMF642-compliant normalised alarm.

    Mandatory fields follow TMF642 §4 "Alarm" resource schema.
    Extended fields (prefixed with x_) carry useful context beyond
    the spec but are kept separate to preserve interoperability.
    """

    # ── TMF642 Mandatory ──────────────────────────────
    id: str                          # internal UUID assigned at normalisation
    alarm_raised_time: datetime      # time alarm first raised at source
    alarm_reporting_time: datetime   # time this record was ingested/reported
    alarm_type: AlarmType
    perceived_severity: PerceivedSeverity
    alarmed_object: AlarmedObjectRef
    probable_cause: str              # ITU-T X.733 / 3GPP TS 32.111-2 Annex B value
    state: AlarmState

    # ── TMF642 Optional ───────────────────────────────
    specific_problem: Optional[str] = None
    alarmed_object_type: Optional[str] = None  # MO class
    alarm_details: Optional[str] = None
    alarm_changed_time: Optional[datetime] = None
    alarm_cleared_time: Optional[datetime] = None
    ack_state: AckState = AckState.UNACKNOWLEDGED
    ack_system_id: Optional[str] = None
    ack_user_id: Optional[str] = None
    service_affecting: bool = False
    alarm_escalation: bool = False
    proposed_repair_actions: Optional[str] = None
    is_root_cause: bool = False

    # ── TMF642 Related objects ────────────────────────
    affected_service: List[ServiceRef] = field(default_factory=list)
    parent_alarm: List[AlarmRef] = field(default_factory=list)
    correlated_alarm: List[AlarmRef] = field(default_factory=list)
    comments: List[Comment] = field(default_factory=list)
    crossed_threshold_information: Optional[ThresholdCrossedInfo] = None

    # ── Source provenance ─────────────────────────────
    external_alarm_id: Optional[str] = None   # original ID in source system
    source_system_id: Optional[str] = None    # NMS/EMS that reported it
    reporting_system_id: Optional[str] = None # forwarding mediation system

    # ── Extension — multi-domain ──────────────────────
    x_domain: NetworkDomain = NetworkDomain.UNKNOWN
    x_vendor: Optional[str] = None
    x_raw_format: Optional[str] = None   # "snmp_trap" | "syslog" | "restconf" | etc.
    x_raw_payload: Optional[Dict[str, Any]] = None  # original parsed payload (for audit)
    x_normalisation_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to TMF642-compatible dict (camelCase keys)"""
        d: Dict[str, Any] = {
            "@type": "Alarm",
            "id": self.id,
            "alarmRaisedTime": self.alarm_raised_time.isoformat() + "Z",
            "alarmReportingTime": self.alarm_reporting_time.isoformat() + "Z",
            "alarmType": self.alarm_type.value,
            "perceivedSeverity": self.perceived_severity.value,
            "alarmedObject": {
                "@type": "AlarmedObjectRef",
                "id": self.alarmed_object.id,
                **({"href": self.alarmed_object.href} if self.alarmed_object.href else {}),
                **({"name": self.alarmed_object.name} if self.alarmed_object.name else {}),
                **({"@referredType": self.alarmed_object.referred_type} if self.alarmed_object.referred_type else {}),
            },
            "probableCause": self.probable_cause,
            "state": self.state.value,
            "ackState": self.ack_state.value,
            "serviceAffecting": self.service_affecting,
            "alarmEscalation": self.alarm_escalation,
            "isRootCause": self.is_root_cause,
        }
        # Optional fields — only include if set
        if self.specific_problem:
            d["specificProblem"] = self.specific_problem
        if self.alarmed_object_type:
            d["alarmedObjectType"] = self.alarmed_object_type
        if self.alarm_details:
            d["alarmDetails"] = self.alarm_details
        if self.alarm_changed_time:
            d["alarmChangedTime"] = self.alarm_changed_time.isoformat() + "Z"
        if self.alarm_cleared_time:
            d["alarmClearedTime"] = self.alarm_cleared_time.isoformat() + "Z"
        if self.proposed_repair_actions:
            d["proposedRepairActions"] = self.proposed_repair_actions
        if self.affected_service:
            d["affectedService"] = [{"@type": "ServiceRef", "id": s.id} for s in self.affected_service]
        if self.external_alarm_id:
            d["externalAlarmId"] = self.external_alarm_id
        if self.source_system_id:
            d["sourceSystemId"] = self.source_system_id
        if self.reporting_system_id:
            d["reportingSystemId"] = self.reporting_system_id
        if self.crossed_threshold_information:
            cti = self.crossed_threshold_information
            d["crossedThresholdInformation"] = {
                "indicatorName": cti.threshold_indicator,
                "observedValue": str(cti.observed_value),
                "thresholdValue": str(cti.threshold_value),
                "direction": cti.direction,
                **({"granularity": cti.granularity} if cti.granularity else {}),
                **({"indicatorUnit": cti.indicator_unit} if cti.indicator_unit else {}),
            }
        # Extension block
        d["x_extensions"] = {
            "domain": self.x_domain.value,
            **({"vendor": self.x_vendor} if self.x_vendor else {}),
            **({"rawFormat": self.x_raw_format} if self.x_raw_format else {}),
            "normalisationVersion": self.x_normalisation_version,
        }
        return d
