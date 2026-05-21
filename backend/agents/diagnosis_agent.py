"""
DiagnosisAgent
==============
Given a list of Alert objects from MonitorAgent, queries Neo4j for related
alarms, correlates them, runs an LLM root-cause analysis, and returns a
structured Diagnosis.
"""
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from neo4j import Driver

import db
from mcp.tools import (
    CorrelateAlarmsRequest,
    GetRcaRequest,
    correlate_alarms,
    get_rca,
)

if TYPE_CHECKING:
    from agents.monitor_agent import Alert

logger = logging.getLogger(__name__)


@dataclass
class Diagnosis:
    incident_id:        str
    rca_text:           str
    affected_cells:     list[str]
    propagation_path:   list[dict]
    recommended_action: str
    confidence:         str
    timestamp:          str
    alarm_groups:       list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "incident_id":        self.incident_id,
            "rca_text":           self.rca_text,
            "affected_cells":     self.affected_cells,
            "propagation_path":   self.propagation_path,
            "recommended_action": self.recommended_action,
            "confidence":         self.confidence,
            "timestamp":          self.timestamp,
            "alarm_groups":       self.alarm_groups,
        }


class DiagnosisAgent:
    """
    Synchronous (blocking) — run from a ThreadPoolExecutor, not the event loop.
    """

    def __init__(self) -> None:
        self._diagnosed_incidents: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def _incident_key(anomalies: list) -> str:
        return "|".join(sorted({a.cell_id for a in anomalies}))

    def diagnose(self, anomalies: list) -> "Optional[Diagnosis]":
        """
        anomalies: list of Alert (objects with .cell_id and .gnb_id attributes).
        Returns a Diagnosis dataclass.
        """
        incident_key = self._incident_key(anomalies)
        with self._lock:
            if incident_key in self._diagnosed_incidents:
                logger.info(
                    "DiagnosisAgent: incident already diagnosed for cells=%s — skipping",
                    [a.cell_id for a in anomalies],
                )
                return None

        driver: Driver = db.get_driver()
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        cell_ids = [a.cell_id for a in anomalies]
        gnb_ids  = list({a.gnb_id for a in anomalies})

        alarm_ids = self._fetch_alarm_ids(driver, cell_ids, gnb_ids)

        corr_result = correlate_alarms(
            body=CorrelateAlarmsRequest(alarm_ids=alarm_ids),
            driver=driver,
        )
        groups = corr_result.get("groups", [])

        correlated_ids = list({
            a["id"]
            for g in groups
            for a in g.get("alarms", [])
            if a.get("id")
        })

        try:
            rca_result = get_rca(
                body=GetRcaRequest(
                    incident_id=incident_id,
                    anomaly_ids=cell_ids,
                    alarm_ids=correlated_ids or alarm_ids,
                ),
                driver=driver,
            )
        except Exception as exc:
            from fastapi import HTTPException as _HTTPException
            if isinstance(exc, _HTTPException) and exc.status_code == 503:
                rca_text = (
                    "LLM not configured — set OPENAI_API_KEY to enable AI diagnosis"
                )
                logger.warning("DiagnosisAgent: LLM not configured — returning fallback")
            else:
                rca_text = "LLM unavailable; manual investigation required."
                logger.exception("DiagnosisAgent: get_rca failed — using fallback")
            rca_result = {
                "rca_text":           rca_text,
                "recommended_action": "Check affected cells and related alarms manually.",
                "confidence":         "low",
                "propagation_path":   [],
            }

        with self._lock:
            self._diagnosed_incidents.add(incident_key)

        return Diagnosis(
            incident_id        = incident_id,
            rca_text           = rca_result.get("rca_text", ""),
            affected_cells     = cell_ids,
            propagation_path   = rca_result.get("propagation_path", []),
            recommended_action = rca_result.get("recommended_action", ""),
            confidence         = rca_result.get("confidence", "medium"),
            timestamp          = ts,
            alarm_groups       = groups,
        )

    def _fetch_alarm_ids(
        self, driver: Driver, cell_ids: list[str], gnb_ids: list[str]
    ) -> list[str]:
        """Return IDs of raised alarms triggered on the affected cells / gNBs."""
        with driver.session() as session:
            rows = session.run(
                "MATCH (a:Alarm {state: 'raised'})-[:TRIGGERED_ON]->(n) "
                "WHERE n.id IN $ids "
                "RETURN a.id AS alarm_id",
                ids=cell_ids + gnb_ids,
            ).data()
        ids = [r["alarm_id"] for r in rows if r["alarm_id"]]
        if not ids:
            logger.info(
                "DiagnosisAgent: no raised alarms for cells=%s gnbs=%s",
                cell_ids, gnb_ids,
            )
        return ids
