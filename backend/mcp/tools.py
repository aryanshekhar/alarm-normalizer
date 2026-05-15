from typing import Optional

from fastapi import APIRouter, Depends, Query
from neo4j import Driver

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
            label_union = " | ".join(f":{lbl}" for lbl in labels)
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


@router.post("/train_model")
def train_model() -> dict:
    return {"status": "ok", "tool": "train_model"}


@router.post("/run_inference")
def run_inference() -> dict:
    return {"status": "ok", "tool": "run_inference"}


@router.post("/correlate_alarms")
def correlate_alarms() -> dict:
    return {"status": "ok", "tool": "correlate_alarms"}


@router.post("/get_rca")
def get_rca() -> dict:
    return {"status": "ok", "tool": "get_rca"}


@router.post("/ask_assistant")
def ask_assistant() -> dict:
    return {"status": "ok", "tool": "ask_assistant"}
