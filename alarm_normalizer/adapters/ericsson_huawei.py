"""
Ericsson ENM (Ericsson Network Manager) Fault Management Adapter
Huawei iManager U2000/M2000 Adapter

Ericsson ENM exports alarms via REST API or JMS/SNMP northbound.
Huawei iManager exports via SNMP traps or northbound REST/CORBA.
"""
from __future__ import annotations
from typing import Any, Dict, Optional

from core.base_adapter import BaseAdapter, map_severity, map_probable_cause, infer_alarm_type
from core.model import (
    CanonicalAlarm, AlarmedObjectRef, AlarmState, AlarmType,
    PerceivedSeverity, NetworkDomain
)

# ─────────────────────────────────────────────────────────────────────────────
# Ericsson ENM
# ─────────────────────────────────────────────────────────────────────────────
# Sample Ericsson ENM alarm JSON (from ENM REST /alarms endpoint):
# {
#   "alarmId": 20240115001234,
#   "objectOfFailure": "SubNetwork=Europe,MeContext=RBS-SITE-001,ManagedElement=1,ENodeBFunction=1",
#   "specificProblem": "Cell Disabled",
#   "probableCause": "CELL_DISABLED",
#   "perceivedSeverity": "A1",           # Ericsson: A1=Critical, A2=Major, A3=Minor, A4=Warning
#   "alarmText": "E-UTRAN cell CELL-001-A has been disabled",
#   "eventTime": "2024-01-15T14:23:01.456Z",
#   "ceaseTime": null,
#   "alarmingObject": "EUtranCellFDD=CELL-001-A",
#   "alarmingObjectType": "EUtranCellFDD",
#   "mangedObjectType": "ENodeBFunction",
#   "insertTime": "2024-01-15T14:23:05.000Z",
#   "ackStatus": "UNACKNOWLEDGED",
#   "serviceAffecting": "SA"
# }

_ENM_SEVERITY_MAP = {
    "A1": PerceivedSeverity.CRITICAL,
    "A2": PerceivedSeverity.MAJOR,
    "A3": PerceivedSeverity.MINOR,
    "A4": PerceivedSeverity.WARNING,
    "CLEARED": PerceivedSeverity.CLEARED,
    "INDETERMINATE": PerceivedSeverity.INDETERMINATE,
    # ENM also sometimes uses full string names
    "CRITICAL": PerceivedSeverity.CRITICAL,
    "MAJOR": PerceivedSeverity.MAJOR,
    "MINOR": PerceivedSeverity.MINOR,
    "WARNING": PerceivedSeverity.WARNING,
}


class EricssonENMAdapter(BaseAdapter):
    """Normalises Ericsson ENM fault management alarms (RAN/Core)."""
    VENDOR = "Ericsson"
    DOMAIN = NetworkDomain.RAN
    RAW_FORMAT = "json_rest"

    def parse(self, raw: Dict) -> CanonicalAlarm:
        sev_raw = str(raw.get("perceivedSeverity", "INDETERMINATE")).upper()
        severity = _ENM_SEVERITY_MAP.get(sev_raw, PerceivedSeverity.INDETERMINATE)

        pc_raw = raw.get("probableCause", "")
        probable_cause = map_probable_cause(pc_raw)

        # Extract NE name from Ericsson 3GPP DN
        dn = raw.get("objectOfFailure", "")
        ne_name = self._extract_ne_name(dn) or "unknown-enb"

        ceased = bool(raw.get("ceaseTime"))
        state = AlarmState.CLEARED if ceased else AlarmState.RAISED
        if ceased:
            severity = PerceivedSeverity.CLEARED

        mo_type = raw.get("alarmingObjectType") or raw.get("mangedObjectType", "RAN-NE")
        alarm_obj_id = raw.get("alarmingObject") or ne_name

        alarm = CanonicalAlarm(
            id=self._new_id(),
            alarm_raised_time=self._parse_ts(raw.get("eventTime", "")),
            alarm_reporting_time=self._parse_ts(raw.get("insertTime", "")) or self._now(),
            alarm_type=infer_alarm_type(probable_cause, mo_type,
                                        raw.get("specificProblem", "")),
            perceived_severity=severity,
            state=state,
            alarmed_object=AlarmedObjectRef(
                id=alarm_obj_id,
                name=ne_name,
                referred_type=mo_type,
            ),
            probable_cause=probable_cause,
            specific_problem=raw.get("specificProblem"),
            alarm_details=raw.get("alarmText"),
            alarmed_object_type=mo_type,
            service_affecting=(raw.get("serviceAffecting", "").upper() == "SA"),
            ack_state=__import__("core.model", fromlist=["AckState"]).AckState.ACKNOWLEDGED
                if raw.get("ackStatus", "").upper() == "ACKNOWLEDGED"
                else __import__("core.model", fromlist=["AckState"]).AckState.UNACKNOWLEDGED,
            source_system_id="Ericsson-ENM",
            external_alarm_id=str(raw.get("alarmId", "")),
        )
        if ceased:
            alarm.alarm_cleared_time = self._parse_ts(raw["ceaseTime"])
        return alarm

    @staticmethod
    def _extract_ne_name(dn: str) -> Optional[str]:
        """Extract MeContext name from Ericsson 3GPP Distinguished Name."""
        if not dn:
            return None
        for part in dn.split(","):
            if part.strip().startswith("MeContext="):
                return part.strip().split("=", 1)[1]
            if part.strip().startswith("ManagedElement="):
                # Fallback to ManagedElement if no MeContext
                pass
        return dn.split(",")[0].split("=")[-1] if "=" in dn else dn


# ─────────────────────────────────────────────────────────────────────────────
# Huawei iManager U2000 / M2000
# ─────────────────────────────────────────────────────────────────────────────
# Huawei alarms arrive as SNMP traps or iManager northbound JSON.
# Sample iManager JSON (U2000 northbound REST alarm notification):
# {
#   "alarmId": "HW-2024011500056789",
#   "deviceName": "HW-AGG-RTR-01",
#   "deviceIp": "10.1.1.101",
#   "alarmName": "Interface Down",
#   "alarmLevel": "Critical",            # Critical/Major/Minor/Warning/Cleared
#   "alarmCategory": "Equipment Alarm",
#   "alarmSource": "GigabitEthernet0/0/1",
#   "alarmReason": "The interface status changes to Down.",
#   "alarmTime": "2024-01-15 14:23:01",
#   "clearTime": null,
#   "ackTime": null,
#   "alarmCode": "ALM-3276",
#   "locationInfo": "DC1-RACK-A3",
#   "additionalInfo": "ifIndex=5; ifDescr=GigabitEthernet0/0/1",
#   "nmsId": "U2000-PROD-01"
# }

_HUAWEI_SEVERITY_MAP = {
    "Emergency": PerceivedSeverity.CRITICAL,
    "Alert": PerceivedSeverity.CRITICAL,
    "Critical": PerceivedSeverity.CRITICAL,
    "Major": PerceivedSeverity.MAJOR,
    "Minor": PerceivedSeverity.MINOR,
    "Warning": PerceivedSeverity.WARNING,
    "Indeterminate": PerceivedSeverity.INDETERMINATE,
    "Cleared": PerceivedSeverity.CLEARED,
}

_HUAWEI_CAT_MAP = {
    "Equipment Alarm": AlarmType.EQUIPMENT_ALARM,
    "Communication Alarm": AlarmType.COMMUNICATIONS_ALARM,
    "Processing Error Alarm": AlarmType.PROCESSING_ERROR_ALARM,
    "Quality of Service Alarm": AlarmType.QUALITY_OF_SERVICE_ALARM,
    "Environmental Alarm": AlarmType.ENVIRONMENTAL_ALARM,
}


class HuaweiIManagerAdapter(BaseAdapter):
    """Normalises Huawei iManager U2000/M2000 alarm JSON."""
    VENDOR = "Huawei"
    DOMAIN = NetworkDomain.IP  # Override per-instance if RAN
    RAW_FORMAT = "json_rest"

    def parse(self, raw: Dict) -> CanonicalAlarm:
        sev_raw = raw.get("alarmLevel", "Indeterminate")
        severity = _HUAWEI_SEVERITY_MAP.get(sev_raw, PerceivedSeverity.INDETERMINATE)

        alarm_name = raw.get("alarmName", "")
        pc_raw = alarm_name.upper().replace(" ", "_")
        probable_cause = map_probable_cause(pc_raw) or map_probable_cause(alarm_name)

        cat_raw = raw.get("alarmCategory", "")
        alarm_type = _HUAWEI_CAT_MAP.get(cat_raw, infer_alarm_type(probable_cause))

        device = raw.get("deviceName", raw.get("deviceIp", "unknown-device"))
        cleared = bool(raw.get("clearTime"))
        state = AlarmState.CLEARED if cleared else AlarmState.RAISED
        if cleared:
            severity = PerceivedSeverity.CLEARED

        return CanonicalAlarm(
            id=self._new_id(),
            alarm_raised_time=self._parse_ts(raw.get("alarmTime", "")),
            alarm_reporting_time=self._now(),
            alarm_type=alarm_type,
            perceived_severity=severity,
            state=state,
            alarmed_object=AlarmedObjectRef(
                id=device,
                name=device,
                referred_type="Router"
            ),
            probable_cause=probable_cause,
            specific_problem=alarm_name,
            alarm_details=raw.get("alarmReason") or raw.get("additionalInfo"),
            alarmed_object_type=raw.get("alarmSource", "Interface"),
            service_affecting=severity in (PerceivedSeverity.CRITICAL, PerceivedSeverity.MAJOR),
            source_system_id=raw.get("nmsId", "Huawei-iManager"),
            external_alarm_id=str(raw.get("alarmId", "")),
            alarm_cleared_time=self._parse_ts(raw["clearTime"]) if cleared else None,
            proposed_repair_actions=self._suggest_repair(alarm_name),
        )

    @staticmethod
    def _suggest_repair(alarm_name: str) -> Optional[str]:
        m = {
            "Interface Down": "Check physical cable, SFP module; verify remote-end configuration",
            "BGP Peer Down": "Verify BGP neighbour reachability, check AS numbers and authentication",
            "CPU Utilization High": "Identify high-CPU processes; check for routing storms or DDoS",
            "Memory Utilization High": "Identify memory-intensive processes; consider reloading process",
            "Fan Failure": "Replace faulty fan tray; check airflow and ambient temperature",
        }
        return m.get(alarm_name)
