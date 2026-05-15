from typing import Optional

from fastapi import APIRouter, Depends, Query
from neo4j import Driver
from pydantic import BaseModel

import db

router = APIRouter(prefix="/tools", tags=["tools"])

# Map user-facing domain name → Neo4j labels to query
_DOMAIN_LABELS: dict[str, list[str]] = {
    "optical": ["OpticalNode"],
    "ip":      ["IPNode"],
    "ran":     ["RANNode", "Cell"],
    "compute": ["Host", "VNF"],
    "service": ["Service", "NetworkSlice"],
    "alarm":   ["Alarm"],
}

# All topology labels (excludes alarms/services by default)
_TOPO_LABELS = ["OpticalNode", "IPNode", "RANNode", "Cell", "Host", "VNF",
                "Service", "NetworkSlice"]


# ─────────────────────────────────────────────────────────────────────────────
# GET /tools/get_topology
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/get_topology")
def get_topology(
    domain: Optional[str] = Query(
        None,
        description="Domain filter: optical | ip | ran | compute | service | alarm. "
                    "Omit to return the full topology.",
    ),
    driver: Driver = Depends(db.get_driver),
) -> dict:
    labels = _DOMAIN_LABELS.get(domain) if domain else _TOPO_LABELS

    with driver.session() as session:
        if labels:
            node_rows = session.run(
                f"MATCH (n) WHERE n:{' OR n:'.join(labels)} "
                "RETURN labels(n) AS labels, n.id AS id, properties(n) AS props"
            ).data()
            node_ids = [r["id"] for r in node_rows if r["id"] is not None]
            edge_rows = session.run(
                "MATCH (a)-[r]->(b) "
                "WHERE a.id IN $ids AND b.id IN $ids "
                "RETURN type(r) AS type, a.id AS from, b.id AS to, properties(r) AS props",
                ids=node_ids,
            ).data()
        else:
            node_rows = session.run(
                "MATCH (n) WHERE n.id IS NOT NULL "
                "RETURN labels(n) AS labels, n.id AS id, properties(n) AS props"
            ).data()
            node_ids = [r["id"] for r in node_rows if r["id"] is not None]
            edge_rows = session.run(
                "MATCH (a)-[r]->(b) "
                "WHERE a.id IS NOT NULL AND b.id IS NOT NULL "
                "RETURN type(r) AS type, a.id AS from, b.id AS to, properties(r) AS props"
            ).data()

    nodes = [{"id": r["id"], "labels": r["labels"], **r["props"]} for r in node_rows]
    edges = [{"type": r["type"], "from": r["from"], "to": r["to"], **r["props"]}
             for r in edge_rows]

    return {
        "status": "ok",
        "tool": "get_topology",
        "domain": domain or "all",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /tools/correlate_alarms
# ─────────────────────────────────────────────────────────────────────────────

class CorrelateAlarmsRequest(BaseModel):
    alarm_ids: list[str] = []       # empty → all currently raised alarms
    include_cleared: bool = False   # include alarms with state='cleared'


def _load_alarms(session, alarm_ids: list[str], include_cleared: bool) -> dict:
    """Return {alarm_id: enriched alarm dict} for the requested scope."""
    if alarm_ids:
        rows = session.run(
            "MATCH (a:Alarm) WHERE a.id IN $ids "
            "OPTIONAL MATCH (a)-[:TRIGGERED_ON]->(n) "
            "WITH a, n.id AS triggered_on "
            "OPTIONAL MATCH (a)-[:AFFECTS_SERVICE]->(s:Service) "
            "RETURN a.id AS id, properties(a) AS props, "
            "       triggered_on, collect(s.id) AS services",
            ids=alarm_ids,
        ).data()
    elif include_cleared:
        rows = session.run(
            "MATCH (a:Alarm) "
            "OPTIONAL MATCH (a)-[:TRIGGERED_ON]->(n) "
            "WITH a, n.id AS triggered_on "
            "OPTIONAL MATCH (a)-[:AFFECTS_SERVICE]->(s:Service) "
            "RETURN a.id AS id, properties(a) AS props, "
            "       triggered_on, collect(s.id) AS services"
        ).data()
    else:
        rows = session.run(
            "MATCH (a:Alarm {state: 'raised'}) "
            "OPTIONAL MATCH (a)-[:TRIGGERED_ON]->(n) "
            "WITH a, n.id AS triggered_on "
            "OPTIONAL MATCH (a)-[:AFFECTS_SERVICE]->(s:Service) "
            "RETURN a.id AS id, properties(a) AS props, "
            "       triggered_on, collect(s.id) AS services"
        ).data()

    result = {}
    for r in rows:
        aid = r["id"]
        if aid is None:
            continue
        alarm = dict(r["props"])
        alarm["triggered_on"] = r["triggered_on"]
        alarm["affected_services"] = r["services"]
        result[aid] = alarm
    return result


def _propagation_groups(
    session, scope_ids: set[str], alarms_by_id: dict
) -> tuple[list[dict], set[str]]:
    """
    Walk PROPAGATED_TO chains starting from root-cause alarms within scope.
    Returns (groups, set_of_covered_alarm_ids).
    """
    if not scope_ids:
        return [], set()

    rows = session.run(
        "MATCH (root:Alarm {isRootCause: true}) "
        "WHERE root.id IN $ids "
        "OPTIONAL MATCH (root)-[:PROPAGATED_TO*1..10]->(symptom:Alarm) "
        "WHERE symptom.id IN $ids "
        "WITH root, collect(distinct symptom.id) AS symptom_ids "
        "OPTIONAL MATCH (root)-[:TRIGGERED_ON]->(rn) "
        "RETURN root.id AS root_id, symptom_ids, rn.id AS root_node_id",
        ids=list(scope_ids),
    ).data()

    groups: list[dict] = []
    covered: set[str] = set()

    for row in rows:
        root_id = row["root_id"]
        symptom_ids: list[str] = [s for s in (row["symptom_ids"] or []) if s]
        all_ids = [root_id] + symptom_ids
        covered.update(all_ids)

        affected_services: set[str] = set()
        for aid in all_ids:
            affected_services.update(alarms_by_id.get(aid, {}).get("affected_services", []))

        groups.append({
            "group_id": f"PROP-{root_id}",
            "correlation_type": "propagation",
            "root_cause_id": root_id,
            "root_node_id": row["root_node_id"],
            "alarm_count": len(all_ids),
            "alarms": [alarms_by_id[aid] for aid in all_ids if aid in alarms_by_id],
            "affected_services": sorted(affected_services),
        })

    return groups, covered


def _service_groups(
    session, remaining_ids: set[str], alarms_by_id: dict
) -> tuple[list[dict], set[str]]:
    """
    Group remaining alarms that share an AFFECTS_SERVICE edge to the same service.
    Returns (groups, set_of_covered_alarm_ids).
    """
    if not remaining_ids:
        return [], set()

    rows = session.run(
        "MATCH (a:Alarm)-[:AFFECTS_SERVICE]->(s:Service) "
        "WHERE a.id IN $ids "
        "WITH s.id AS service_id, collect(distinct a.id) AS alarm_ids "
        "WHERE size(alarm_ids) > 1 "
        "RETURN service_id, alarm_ids",
        ids=list(remaining_ids),
    ).data()

    groups: list[dict] = []
    covered: set[str] = set()

    for row in rows:
        aids: list[str] = row["alarm_ids"]
        covered.update(aids)
        groups.append({
            "group_id": f"SVC-{row['service_id']}",
            "correlation_type": "service",
            "root_cause_id": None,
            "root_node_id": None,
            "alarm_count": len(aids),
            "alarms": [alarms_by_id[aid] for aid in aids if aid in alarms_by_id],
            "affected_services": [row["service_id"]],
        })

    return groups, covered


def _isolated_groups(isolated_ids: set[str], alarms_by_id: dict) -> list[dict]:
    return [
        {
            "group_id": f"ISO-{aid}",
            "correlation_type": "isolated",
            "root_cause_id": None,
            "root_node_id": None,
            "alarm_count": 1,
            "alarms": [alarms_by_id[aid]],
            "affected_services": alarms_by_id[aid].get("affected_services", []),
        }
        for aid in isolated_ids
        if aid in alarms_by_id
    ]


@router.post("/correlate_alarms")
def correlate_alarms(
    body: CorrelateAlarmsRequest = CorrelateAlarmsRequest(),
    driver: Driver = Depends(db.get_driver),
) -> dict:
    with driver.session() as session:
        alarms_by_id = _load_alarms(session, body.alarm_ids, body.include_cleared)
        scope_ids = set(alarms_by_id)

        prop_groups, prop_ids = _propagation_groups(session, scope_ids, alarms_by_id)
        remaining = scope_ids - prop_ids
        svc_groups, svc_ids = _service_groups(session, remaining, alarms_by_id)
        isolated = _isolated_groups(remaining - svc_ids, alarms_by_id)

    groups = prop_groups + svc_groups + isolated

    return {
        "status": "ok",
        "tool": "correlate_alarms",
        "alarm_count": len(scope_ids),
        "group_count": len(groups),
        "groups": groups,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stubs
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/train_model")
def train_model() -> dict:
    return {"status": "ok", "tool": "train_model"}


@router.post("/run_inference")
def run_inference() -> dict:
    return {"status": "ok", "tool": "run_inference"}


@router.post("/get_rca")
def get_rca() -> dict:
    return {"status": "ok", "tool": "get_rca"}


@router.post("/ask_assistant")
def ask_assistant() -> dict:
    return {"status": "ok", "tool": "ask_assistant"}
