"""
Nokia NetAct FM / Nokia 1830 PSS Adapter

Nokia Network Management systems (NetAct for RAN/Core, 1830 PSS for Optical)
export alarms as JSON payloads via RESTCONF or northbound FTP bulk files.

Nokia RAN alarm structure (NetAct):
{
  "notificationId": "12345678",
  "eventTime": "2024-01-15T14:23:01+00:00",
  "alarmType": "EQUIPMENT_ALARM",
  "probableCause": "RRH_FAULT",
  "perceivedSeverity": "MAJOR",
  "specificProblem": "Radio Unit Hardware Fault",
  "managedObjectClass": "BTS",
  "managedObjectInstance": "PLMN-PLMN/BSC-1/BTS-101",
  "additionalText": "RRH unit temperature exceeded 85°C",
  "alarmId": "BTS-101-RRH-FAULT-20240115",
  "neId": "BTS-101",
  "neName": "SITE-ALPHA-BTS1",
  "vendor": "Nokia",
  "domain": "RAN",
  "clearingTime": null,
  "proposedRepairAction": "Replace RRH unit or check cooling"
}

Nokia 1830 PSS (Optical) structure:
{
  "alarmSequenceNo": 9876,
  "raisedTime": "2024-01-15T14:22:45Z",
  "alarmSeverity": "CRITICAL",
  "alarmCondition": "LOS",
  "serviceAffecting": true,
  "neName": "PSS-32-NODE-A",
  "neType": "1830PSS32",
  "facilityType": "OCH",
  "facilityName": "OCH-1-1-1-TX",
  "conditionDescription": "Loss of Signal on Optical Channel",
  "additionalInfo": "OSNR: -3.2 dB",
  "clearTime": null
}
"""
from __future__ import annotations
from typing import Any, Dict, Optional

from core.base_adapter import BaseAdapter, map_severity, map_probable_cause, infer_alarm_type
from core.model import (
    CanonicalAlarm, AlarmedObjectRef, AlarmState, AlarmType,
    PerceivedSeverity, NetworkDomain, ServiceRef, ThresholdCrossedInfo
)

_NOKIA_ALARM_TYPE_MAP = {
    "COMMUNICATIONS_ALARM": AlarmType.COMMUNICATIONS_ALARM,
    "EQUIPMENT_ALARM": AlarmType.EQUIPMENT_ALARM,
    "PROCESSING_ERROR_ALARM": AlarmType.PROCESSING_ERROR_ALARM,
    "ENVIRONMENTAL_ALARM": AlarmType.ENVIRONMENTAL_ALARM,
    "QUALITY_OF_SERVICE_ALARM": AlarmType.QUALITY_OF_SERVICE_ALARM,
}


class NokiaNetActAdapter(BaseAdapter):
    """
    Normalises Nokia NetAct FM JSON alarms (RAN / Core domains).
    """
    VENDOR = "Nokia"
    DOMAIN = NetworkDomain.RAN
    RAW_FORMAT = "json_restconf"

    def parse(self, raw: Dict) -> CanonicalAlarm:
        severity  = map_severity(raw.get("perceivedSeverity", "INDETERMINATE"))
        pc_raw    = raw.get("probableCause", "UNKNOWN")
        probable_cause = map_probable_cause(pc_raw)

        at_raw = raw.get("alarmType", "")
        alarm_type = _NOKIA_ALARM_TYPE_MAP.get(
            at_raw.upper().replace(" ", "_"),
            infer_alarm_type(probable_cause)
        )

        ne_id   = raw.get("neId") or raw.get("managedObjectInstance", "unknown")
        ne_name = raw.get("neName", ne_id)
        mo_class = raw.get("managedObjectClass", "NetworkElement")

        cleared = bool(raw.get("clearingTime") or raw.get("clearTime"))
        state = AlarmState.CLEARED if cleared else AlarmState.RAISED
        if cleared:
            severity = PerceivedSeverity.CLEARED

        alarm = CanonicalAlarm(
            id=self._new_id(),
            alarm_raised_time=self._parse_ts(raw.get("eventTime") or raw.get("raisedTime", "")),
            alarm_reporting_time=self._now(),
            alarm_type=alarm_type,
            perceived_severity=severity,
            state=state,
            alarmed_object=AlarmedObjectRef(
                id=ne_id,
                name=ne_name,
                referred_type=mo_class,
            ),
            probable_cause=probable_cause,
            specific_problem=raw.get("specificProblem") or raw.get("alarmCondition"),
            alarm_details=raw.get("additionalText") or raw.get("conditionDescription")
                         or raw.get("additionalInfo"),
            alarmed_object_type=mo_class,
            service_affecting=bool(raw.get("serviceAffecting", False)),
            source_system_id="Nokia-NetAct",
            external_alarm_id=str(raw.get("alarmId") or raw.get("notificationId")
                                  or raw.get("alarmSequenceNo", "")),
            proposed_repair_actions=raw.get("proposedRepairAction"),
        )
        if cleared and raw.get("clearingTime"):
            alarm.alarm_cleared_time = self._parse_ts(raw["clearingTime"])
        return alarm


class Nokia1830PSSAdapter(BaseAdapter):
    """
    Normalises Nokia 1830 PSS optical alarms.
    Same parent class — different DOMAIN and field mapping.
    """
    VENDOR = "Nokia"
    DOMAIN = NetworkDomain.OPTICAL
    RAW_FORMAT = "json_restconf"

    def parse(self, raw: Dict) -> CanonicalAlarm:
        severity = map_severity(raw.get("alarmSeverity", "INDETERMINATE"))
        pc_raw   = raw.get("alarmCondition", "UNKNOWN")
        probable_cause = map_probable_cause(pc_raw)

        cleared = bool(raw.get("clearTime"))
        state = AlarmState.CLEARED if cleared else AlarmState.RAISED
        if cleared:
            severity = PerceivedSeverity.CLEARED

        ne_name = raw.get("neName", "unknown-optical-ne")
        facility = raw.get("facilityName", "")
        obj_id = f"{ne_name}/{facility}" if facility else ne_name

        alarm = CanonicalAlarm(
            id=self._new_id(),
            alarm_raised_time=self._parse_ts(raw.get("raisedTime", "")),
            alarm_reporting_time=self._now(),
            alarm_type=infer_alarm_type(probable_cause, raw.get("neType", "OTN")),
            perceived_severity=severity,
            state=state,
            alarmed_object=AlarmedObjectRef(
                id=obj_id,
                name=ne_name,
                referred_type=raw.get("neType", "OpticalNE"),
            ),
            probable_cause=probable_cause,
            specific_problem=raw.get("alarmCondition"),
            alarm_details=raw.get("conditionDescription") or raw.get("additionalInfo"),
            alarmed_object_type=raw.get("facilityType", "OpticalFacility"),
            service_affecting=bool(raw.get("serviceAffecting", True)),
            source_system_id="Nokia-1830PSS",
            external_alarm_id=str(raw.get("alarmSequenceNo", "")),
        )
        if cleared:
            alarm.alarm_cleared_time = self._parse_ts(raw["clearTime"])
        return alarm
