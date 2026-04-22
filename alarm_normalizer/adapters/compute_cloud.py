"""
Generic SNMP Trap Adapter
Compute Infrastructure Adapter (OpenStack Ceilometer / Kubernetes Events / Prometheus AlertManager)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from core.base_adapter import BaseAdapter, map_severity, map_probable_cause, infer_alarm_type
from core.model import (
    CanonicalAlarm, AlarmedObjectRef, AlarmState, AlarmType,
    PerceivedSeverity, NetworkDomain, ThresholdCrossedInfo
)

# ─────────────────────────────────────────────────────────────────────────────
# Generic SNMP Trap
# ─────────────────────────────────────────────────────────────────────────────
# Input format: dict with varbinds as parsed by pysnmp or net-snmp trap handler
# {
#   "source_ip": "10.1.2.3",
#   "community": "public",
#   "enterprise_oid": "1.3.6.1.6.3.1.1.5.3",  # linkDown
#   "trap_type": "linkDown",
#   "agent_address": "10.1.2.3",
#   "timestamp": "2024-01-15T14:23:01Z",
#   "varbinds": {
#     "sysName": "core-router-1",
#     "ifIndex": "3",
#     "ifDescr": "GigabitEthernet1/0",
#     "ifAdminStatus": "1",
#     "ifOperStatus": "2",
#     "ifAlias": "TO-AGG-SWITCH"
#   }
# }

_SNMP_TRAP_SEVERITY = {
    "coldStart": PerceivedSeverity.MAJOR,
    "warmStart": PerceivedSeverity.WARNING,
    "linkDown": PerceivedSeverity.MAJOR,
    "linkUp": PerceivedSeverity.CLEARED,
    "authenticationFailure": PerceivedSeverity.MINOR,
    "egpNeighborLoss": PerceivedSeverity.MAJOR,
    "enterpriseSpecific": PerceivedSeverity.INDETERMINATE,
}

_SNMP_TRAP_PC = {
    "coldStart": "softwareProgramAbnormallyTerminated",
    "warmStart": "softwareProgramAbnormallyTerminated",
    "linkDown": "communicationsSubsystemFailure",
    "linkUp": "communicationsSubsystemFailure",
    "authenticationFailure": "unauthorizedAccessAttempt",
    "egpNeighborLoss": "softwareProgramAbnormallyTerminated",
}


class SNMPTrapAdapter(BaseAdapter):
    """
    Normalises generic SNMP v1/v2c traps (IF-MIB::linkDown, etc.)
    Works for any device that sends standard IF-MIB, ENTITY-MIB traps.
    """
    VENDOR = "Generic"
    DOMAIN = NetworkDomain.IP
    RAW_FORMAT = "snmp_trap"

    def parse(self, raw: Dict) -> CanonicalAlarm:
        trap_type  = raw.get("trap_type", "enterpriseSpecific")
        varbinds   = raw.get("varbinds", {})
        source_ip  = raw.get("source_ip", raw.get("agent_address", "0.0.0.0"))
        device     = varbinds.get("sysName", source_ip)
        iface      = varbinds.get("ifDescr") or varbinds.get("ifAlias", "")
        timestamp  = raw.get("timestamp", "")

        severity = _SNMP_TRAP_SEVERITY.get(trap_type, PerceivedSeverity.INDETERMINATE)
        pc = _SNMP_TRAP_PC.get(trap_type, "unspecified")

        state = AlarmState.CLEARED if trap_type == "linkUp" else AlarmState.RAISED

        obj_id = f"{device}/{iface}" if iface else device
        details = f"OID={raw.get('enterprise_oid','')}; varbinds={varbinds}"

        return CanonicalAlarm(
            id=self._new_id(),
            alarm_raised_time=self._parse_ts(timestamp) if timestamp else self._now(),
            alarm_reporting_time=self._now(),
            alarm_type=infer_alarm_type(pc, "Router"),
            perceived_severity=severity,
            state=state,
            alarmed_object=AlarmedObjectRef(
                id=obj_id,
                name=device,
                referred_type="NetworkDevice"
            ),
            probable_cause=pc,
            specific_problem=trap_type,
            alarm_details=details,
            alarmed_object_type="Interface" if iface else "NetworkDevice",
            source_system_id=f"snmp-trap-{source_ip}",
            external_alarm_id=f"{source_ip}-{trap_type}-{timestamp}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Compute / Cloud Infrastructure
# ─────────────────────────────────────────────────────────────────────────────

class PrometheusAlertAdapter(BaseAdapter):
    """
    Normalises Prometheus AlertManager webhook payloads.
    AlertManager POST format:
    {
      "alerts": [{
        "status": "firing",
        "labels": {
          "alertname": "HostDown",
          "severity": "critical",
          "instance": "compute-node-07:9100",
          "job": "node_exporter",
          "datacenter": "DC1"
        },
        "annotations": {
          "summary": "Host compute-node-07 is unreachable",
          "description": "compute-node-07 has been down for more than 5 minutes"
        },
        "startsAt": "2024-01-15T14:23:01.000Z",
        "endsAt": "0001-01-01T00:00:00Z",
        "generatorURL": "http://prometheus:9090/graph?...",
        "fingerprint": "abc123def456"
      }]
    }
    """
    VENDOR = "Prometheus"
    DOMAIN = NetworkDomain.COMPUTE
    RAW_FORMAT = "webhook_json"

    def parse(self, raw: Dict) -> CanonicalAlarm:
        # Accept both single alert and AlertManager webhook envelope
        if "alerts" in raw:
            alert = raw["alerts"][0]
        else:
            alert = raw

        labels      = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        status      = alert.get("status", "firing")

        alert_name = labels.get("alertname", "Unknown")
        sev_raw    = labels.get("severity", "warning")
        severity   = map_severity(sev_raw)
        instance   = labels.get("instance", "unknown-host").split(":")[0]

        if status == "resolved":
            state    = AlarmState.CLEARED
            severity = PerceivedSeverity.CLEARED
        else:
            state = AlarmState.RAISED

        pc = map_probable_cause(alert_name.upper().replace(" ", "_"))

        alarm = CanonicalAlarm(
            id=self._new_id(),
            alarm_raised_time=self._parse_ts(alert.get("startsAt", "")),
            alarm_reporting_time=self._now(),
            alarm_type=infer_alarm_type(pc, "ComputeHost"),
            perceived_severity=severity,
            state=state,
            alarmed_object=AlarmedObjectRef(
                id=instance,
                name=instance,
                referred_type="ComputeHost"
            ),
            probable_cause=pc,
            specific_problem=alert_name,
            alarm_details=annotations.get("description") or annotations.get("summary"),
            alarmed_object_type="ComputeNode",
            source_system_id="Prometheus-AlertManager",
            external_alarm_id=alert.get("fingerprint", ""),
        )
        ends_at = alert.get("endsAt", "")
        if status == "resolved" and ends_at and "0001" not in ends_at:
            alarm.alarm_cleared_time = self._parse_ts(ends_at)
        return alarm


class KubernetesEventAdapter(BaseAdapter):
    """
    Normalises Kubernetes Warning Events (from kubectl get events / kube-state-metrics).
    {
      "apiVersion": "v1",
      "kind": "Event",
      "metadata": {"name": "my-pod.1234abc", "namespace": "ran-functions"},
      "involvedObject": {"kind": "Pod", "name": "upf-pod-xyz", "namespace": "ran-functions"},
      "reason": "BackOff",
      "message": "Back-off restarting failed container upf in pod upf-pod-xyz",
      "type": "Warning",
      "firstTimestamp": "2024-01-15T14:23:00Z",
      "lastTimestamp": "2024-01-15T14:23:01Z",
      "count": 5,
      "source": {"component": "kubelet", "host": "worker-node-3"}
    }
    """
    VENDOR = "Kubernetes"
    DOMAIN = NetworkDomain.COMPUTE
    RAW_FORMAT = "k8s_event"

    _REASON_SEVERITY = {
        "OOMKilling": PerceivedSeverity.CRITICAL,
        "BackOff": PerceivedSeverity.MAJOR,
        "Failed": PerceivedSeverity.MAJOR,
        "FailedScheduling": PerceivedSeverity.MAJOR,
        "NodeNotReady": PerceivedSeverity.CRITICAL,
        "NodeNotSchedulable": PerceivedSeverity.MAJOR,
        "Evicted": PerceivedSeverity.MAJOR,
        "Killing": PerceivedSeverity.MINOR,
        "Pulled": PerceivedSeverity.INDETERMINATE,
        "Started": PerceivedSeverity.CLEARED,
    }

    def parse(self, raw: Dict) -> CanonicalAlarm:
        obj     = raw.get("involvedObject", {})
        reason  = raw.get("reason", "Unknown")
        message = raw.get("message", "")
        ns      = obj.get("namespace", "default")
        obj_name = obj.get("name", "unknown")
        obj_kind = obj.get("kind", "Pod")

        severity = self._REASON_SEVERITY.get(reason, PerceivedSeverity.INDETERMINATE)
        pc = map_probable_cause(reason.upper())

        state = AlarmState.CLEARED if severity == PerceivedSeverity.CLEARED else AlarmState.RAISED

        return CanonicalAlarm(
            id=self._new_id(),
            alarm_raised_time=self._parse_ts(raw.get("lastTimestamp", "")),
            alarm_reporting_time=self._now(),
            alarm_type=infer_alarm_type(pc, obj_kind),
            perceived_severity=severity,
            state=state,
            alarmed_object=AlarmedObjectRef(
                id=f"{ns}/{obj_name}",
                name=obj_name,
                referred_type=obj_kind
            ),
            probable_cause=pc,
            specific_problem=reason,
            alarm_details=message,
            alarmed_object_type=obj_kind,
            source_system_id=f"k8s-{raw.get('source',{}).get('host','unknown')}",
            external_alarm_id=raw.get("metadata", {}).get("name", ""),
        )
