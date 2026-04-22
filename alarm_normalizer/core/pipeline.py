"""
Alarm Normalisation Pipeline

Orchestrates adapter selection, normalisation, enrichment, validation,
and output routing.  Designed as a single-pass streaming processor.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

from core.base_adapter import BaseAdapter
from core.model import CanonicalAlarm, NetworkDomain, PerceivedSeverity

# Import all adapters
from adapters.cisco_syslog import CiscoSyslogAdapter
from adapters.nokia_netact import NokiaNetActAdapter, Nokia1830PSSAdapter
from adapters.ericsson_huawei import EricssonENMAdapter, HuaweiIManagerAdapter
from adapters.compute_cloud import SNMPTrapAdapter, PrometheusAlertAdapter, KubernetesEventAdapter

log = logging.getLogger("alarm_pipeline")


# ─────────────────────────────────────────────
# Adapter Registry
# ─────────────────────────────────────────────

class AdapterRegistry:
    """Maps (vendor, domain, format) tuples to adapter classes."""

    _registry: Dict[Tuple, Type[BaseAdapter]] = {
        ("cisco",    "ip",      "syslog"):       CiscoSyslogAdapter,
        ("cisco",    "ip",      "snmp_trap"):    SNMPTrapAdapter,
        ("nokia",    "ran",     "json_restconf"): NokiaNetActAdapter,
        ("nokia",    "optical", "json_restconf"): Nokia1830PSSAdapter,
        ("ericsson", "ran",     "json_rest"):    EricssonENMAdapter,
        ("huawei",   "ip",      "json_rest"):    HuaweiIManagerAdapter,
        ("huawei",   "ran",     "json_rest"):    HuaweiIManagerAdapter,
        ("generic",  "ip",      "snmp_trap"):    SNMPTrapAdapter,
        ("prometheus","compute","webhook_json"): PrometheusAlertAdapter,
        ("kubernetes","compute","k8s_event"):    KubernetesEventAdapter,
    }

    @classmethod
    def get(cls, vendor: str, domain: str, fmt: str) -> Optional[Type[BaseAdapter]]:
        key = (vendor.lower(), domain.lower(), fmt.lower())
        adapter_cls = cls._registry.get(key)
        if adapter_cls:
            return adapter_cls
        # Partial match fallbacks
        for (v, d, f), a in cls._registry.items():
            if v == vendor.lower() and d == domain.lower():
                return a
        for (v, d, f), a in cls._registry.items():
            if v == vendor.lower():
                return a
        return None

    @classmethod
    def register(cls, vendor: str, domain: str, fmt: str,
                 adapter_cls: Type[BaseAdapter]) -> None:
        cls._registry[(vendor.lower(), domain.lower(), fmt.lower())] = adapter_cls


# ─────────────────────────────────────────────
# Pipeline Stages
# ─────────────────────────────────────────────

class ValidationStage:
    """Validates a normalised alarm for mandatory field completeness."""

    REQUIRED_FIELDS = [
        "id", "alarm_raised_time", "alarm_type",
        "perceived_severity", "alarmed_object", "probable_cause", "state"
    ]

    def run(self, alarm: CanonicalAlarm) -> Tuple[bool, List[str]]:
        errors = []
        for f in self.REQUIRED_FIELDS:
            val = getattr(alarm, f, None)
            if val is None or (isinstance(val, str) and not val.strip()):
                errors.append(f"Missing required field: {f}")
        if alarm.alarmed_object and not alarm.alarmed_object.id:
            errors.append("alarmedObject.id is empty")
        return len(errors) == 0, errors


class EnrichmentStage:
    """
    Adds computed/derived fields to the alarm after normalisation.
    In production this would call a CMDB or inventory API.
    """

    # Simple static enrichment lookup (replace with CMDB query in production)
    _DEVICE_META: Dict[str, Dict] = {
        "router-pe1": {"site": "Mumbai-DC1", "region": "south-asia", "rack": "A-12"},
        "gNB-SITE-ALPHA-01": {"site": "Site-Alpha", "region": "south-asia", "sector": "1,2,3"},
        "PSS-32-NODE-A": {"site": "Optical-POP-1", "region": "south-asia", "fiber_route": "Route-7"},
        "compute-node-07": {"site": "MEC-Edge-1", "region": "south-asia", "cluster": "K8S-PROD"},
    }

    def run(self, alarm: CanonicalAlarm) -> CanonicalAlarm:
        ne_id = alarm.alarmed_object.id if alarm.alarmed_object else ""
        # Normalise NE ID for lookup (strip interface suffix)
        base_ne = ne_id.split("/")[0]
        meta = self._DEVICE_META.get(base_ne, {})
        if meta and alarm.alarmed_object:
            # Embed site/region into alarm_details extension
            enrichment = f"[site={meta.get('site','')}; region={meta.get('region','')}]"
            if alarm.alarm_details:
                alarm.alarm_details = alarm.alarm_details + " " + enrichment
            else:
                alarm.alarm_details = enrichment
        return alarm


class DeduplicationStage:
    """
    Simple stateful deduplication — suppresses identical alarms within a window.
    In production this should use a Redis TTL store or similar.
    """
    def __init__(self, window_seconds: int = 300):
        self._seen: Dict[str, datetime] = {}
        self._window = window_seconds

    def is_duplicate(self, alarm: CanonicalAlarm) -> bool:
        key = self._fingerprint(alarm)
        now = datetime.utcnow()
        if key in self._seen:
            age = (now - self._seen[key]).total_seconds()
            if age < self._window:
                log.debug(f"Duplicate suppressed: {key} (age={age:.0f}s)")
                return True
        self._seen[key] = now
        return False

    @staticmethod
    def _fingerprint(alarm: CanonicalAlarm) -> str:
        """Fingerprint based on device + specific problem + severity."""
        obj_id = alarm.alarmed_object.id if alarm.alarmed_object else ""
        return f"{obj_id}::{alarm.specific_problem}::{alarm.perceived_severity.value}"


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

class NormalisationPipeline:
    """
    Single entry point for alarm normalisation.

    Usage:
        pipeline = NormalisationPipeline()
        canonical = pipeline.process(
            raw_payload = {"probableCause": "CELL_DISABLED", ...},
            vendor = "Nokia",
            domain = "ran",
            format = "json_restconf"
        )
    """

    def __init__(self, enable_dedup: bool = True, dedup_window_seconds: int = 300):
        self._validator  = ValidationStage()
        self._enricher   = EnrichmentStage()
        self._deduper    = DeduplicationStage(dedup_window_seconds) if enable_dedup else None
        self._stats      = {"processed": 0, "valid": 0, "invalid": 0,
                            "deduplicated": 0, "by_vendor": {}, "by_domain": {}}

    def process(
        self,
        raw_payload: Any,
        vendor: str,
        domain: str,
        format: str,
    ) -> Optional[CanonicalAlarm]:
        """
        Normalise one raw alarm event.
        Returns a CanonicalAlarm or None if validation failed / duplicate.
        """
        self._stats["processed"] += 1

        # 1. Select adapter
        adapter_cls = AdapterRegistry.get(vendor, domain, format)
        if not adapter_cls:
            log.warning(f"No adapter for vendor={vendor} domain={domain} format={format}")
            self._stats["invalid"] += 1
            return None
        adapter = adapter_cls()

        # 2. Normalise
        try:
            alarm = adapter.normalise(raw_payload)
        except Exception as e:
            log.error(f"Adapter {adapter_cls.__name__} failed: {e}", exc_info=True)
            self._stats["invalid"] += 1
            return None

        # 3. Validate
        valid, errors = self._validator.run(alarm)
        if not valid:
            log.warning(f"Validation failed for {alarm.id}: {errors}")
            self._stats["invalid"] += 1
            return None

        # 4. Deduplication
        if self._deduper and self._deduper.is_duplicate(alarm):
            self._stats["deduplicated"] += 1
            return None

        # 5. Enrich
        alarm = self._enricher.run(alarm)

        # 6. Update stats
        self._stats["valid"] += 1
        v = self._stats["by_vendor"]
        v[vendor] = v.get(vendor, 0) + 1
        d = self._stats["by_domain"]
        d[domain] = d.get(domain, 0) + 1

        return alarm

    def process_batch(
        self,
        events: List[Dict[str, Any]],
    ) -> List[CanonicalAlarm]:
        """
        Process a batch of events.  Each dict must have keys:
          payload, vendor, domain, format
        """
        results = []
        for ev in events:
            alarm = self.process(
                raw_payload=ev["payload"],
                vendor=ev["vendor"],
                domain=ev["domain"],
                format=ev["format"],
            )
            if alarm:
                results.append(alarm)
        return results

    @property
    def stats(self) -> Dict:
        return dict(self._stats)
