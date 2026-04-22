"""
Cisco IOS/IOS-XE Syslog Adapter

Parses structured syslog messages from Cisco routers/switches.
Format: <priority>TIMESTAMP: %FACILITY-SEVERITY-MNEMONIC: message

Examples ingested from Cisco EPN Manager northbound or raw syslog UDP 514.
"""
from __future__ import annotations
import re
from typing import Any, Dict, Optional

from core.base_adapter import BaseAdapter, map_severity, map_probable_cause, infer_alarm_type
from core.model import (
    CanonicalAlarm, AlarmedObjectRef, AlarmState, AlarmType,
    PerceivedSeverity, NetworkDomain
)

# Cisco syslog numeric severity → PerceivedSeverity
_CISCO_SEV = {
    0: PerceivedSeverity.CRITICAL,
    1: PerceivedSeverity.CRITICAL,
    2: PerceivedSeverity.CRITICAL,
    3: PerceivedSeverity.MAJOR,
    4: PerceivedSeverity.MINOR,
    5: PerceivedSeverity.WARNING,
    6: PerceivedSeverity.INDETERMINATE,
    7: PerceivedSeverity.INDETERMINATE,
}

# Regex for Cisco structured syslog
# e.g.: "2024-01-15T14:23:01Z router-pe1 : %LINK-3-UPDOWN: Interface GigabitEthernet0/0/0, changed state to down"
_CISCO_SYSLOG_RE = re.compile(
    r"(?P<facility>[A-Z0-9_]+)-(?P<sev>[0-9])-(?P<mnemonic>[A-Z0-9_]+):\s*(?P<msg>.+)"
)


class CiscoSyslogAdapter(BaseAdapter):
    """
    Normalises Cisco IOS/XE syslog messages (raw string or pre-parsed dict).

    Accepts two input forms:
      1. Raw syslog string (str)
      2. Pre-parsed dict with keys: timestamp, hostname, facility, severity,
         mnemonic, message
    """
    VENDOR = "Cisco"
    DOMAIN = NetworkDomain.IP
    RAW_FORMAT = "syslog"

    def parse(self, raw: Any) -> CanonicalAlarm:
        if isinstance(raw, str):
            data = self._parse_syslog_string(raw)
        else:
            data = raw  # already parsed dict

        facility  = data.get("facility", "UNKNOWN")
        mnemonic  = data.get("mnemonic", "UNKNOWN")
        sev_num   = int(data.get("severity", 6))
        message   = data.get("message", "")
        hostname  = data.get("hostname", "unknown-device")
        timestamp = data.get("timestamp", "")

        alarm_key = f"{facility}-{sev_num}-{mnemonic}"
        probable_cause = map_probable_cause(alarm_key)
        severity = _CISCO_SEV.get(sev_num, PerceivedSeverity.INDETERMINATE)
        alarm_type = infer_alarm_type(probable_cause, "Router", message)

        # Detect cleared state
        state = AlarmState.RAISED
        if any(kw in message.lower() for kw in ("up", "active", "restored", "cleared")):
            if mnemonic in ("UPDOWN", "ADJCHG", "STATECHANGE"):
                if "up" in message.lower() or "established" in message.lower():
                    state = AlarmState.CLEARED
                    severity = PerceivedSeverity.CLEARED

        # Extract interface if present
        iface_match = re.search(
            r"Interface\s+([\w/\.\-]+)", message, re.IGNORECASE
        )
        iface = iface_match.group(1) if iface_match else None

        return CanonicalAlarm(
            id=self._new_id(),
            alarm_raised_time=self._parse_ts(timestamp) if timestamp else self._now(),
            alarm_reporting_time=self._now(),
            alarm_type=alarm_type,
            perceived_severity=severity,
            state=state,
            alarmed_object=AlarmedObjectRef(
                id=hostname,
                name=hostname,
                referred_type="Router"
            ),
            probable_cause=probable_cause,
            specific_problem=f"{facility}-{mnemonic}",
            alarm_details=message,
            alarmed_object_type="Router",
            source_system_id=f"cisco-syslog-{hostname}",
            external_alarm_id=f"{hostname}-{mnemonic}-{timestamp}",
            proposed_repair_actions=self._repair_action(mnemonic, iface),
        )

    def _parse_syslog_string(self, raw: str) -> Dict:
        """Parse a raw Cisco syslog line into structured fields."""
        parts = raw.strip().split()
        hostname = "unknown"
        ts = ""
        rest = raw

        # Try to find hostname and timestamp heuristically
        # Format: <pri>timestamp hostname : %FAC-SEV-MNEM: message
        if "%" in raw:
            percent_idx = raw.index("%")
            before = raw[:percent_idx]
            tokens = before.strip().split()
            if len(tokens) >= 2:
                ts = tokens[0]
                hostname = tokens[1].rstrip(":")
            rest = raw[percent_idx + 1:]

        m = _CISCO_SYSLOG_RE.match(rest)
        if m:
            return {
                "timestamp": ts,
                "hostname": hostname,
                "facility": m.group("facility"),
                "severity": m.group("sev"),
                "mnemonic": m.group("mnemonic"),
                "message": m.group("msg").strip(),
            }
        return {"timestamp": ts, "hostname": hostname, "message": raw, "severity": 6,
                "facility": "UNKNOWN", "mnemonic": "UNKNOWN"}

    @staticmethod
    def _repair_action(mnemonic: str, interface: Optional[str]) -> Optional[str]:
        actions = {
            "UPDOWN": f"Check physical layer on {interface or 'interface'}; verify cable, SFP, remote peer",
            "ADJCHG": "Check BGP/OSPF neighbor reachability; verify MTU, authentication, timers",
            "FAN_FAILURE": "Replace faulty fan module; check chassis airflow",
            "CPU_OVERLOAD": "Check for routing protocol churn or traffic spike; consider rate-limiting",
        }
        return actions.get(mnemonic)
