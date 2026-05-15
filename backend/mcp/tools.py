import asyncio
import json
import threading
from typing import Optional

import torch
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from neo4j import Driver
from pydantic import BaseModel
from torch.utils.data import DataLoader, TensorDataset

import db
import model_store as ms

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
# POST /tools/train_model
# ─────────────────────────────────────────────────────────────────────────────

class TrainModelRequest(BaseModel):
    epochs: int = 15       # training epochs
    data_window: int = 30  # SIMBA sliding window size in seconds


def _do_training(epochs: int, data_window: int, emit) -> None:
    """
    Blocking training function — runs in a thread.
    Calls emit(event_dict) at each SSE milestone.
    Stores the trained model in model_store on completion.
    """
    # Late imports keep module load fast and isolate heavy deps to this thread
    from simba_pipeline.models.simba import Simba, WeightedFocalLoss, compute_class_weights
    from simba_pipeline.data.dataset_generator import (
        create_sliding_windows, train_val_test_split, KPINormalizer,
    )
    from simba_pipeline.training.train import train_one_epoch, evaluate
    from integrated_aiops.scenarios.fault_propagation import IntegratedDatasetGenerator
    from integrated_aiops.topology.unified_topology import build_ran_adjacency

    # ── Stage 1: data preparation ─────────────────────────────────────────────
    emit({"stage": "preparing",
          "message": "Analysing 30 days of network behaviour...",
          "progress": 5})

    duration_s = max(3600, data_window * 200)
    gen     = IntegratedDatasetGenerator(duration_s=duration_s)
    dataset = gen.generate()

    kpi_data = dataset["kpi_data"]   # (T, N_CELLS, N_KPIS)
    labels   = dataset["labels"]     # (T, N_CELLS)
    adj      = build_ran_adjacency() # (N_CELLS, N_CELLS) float32

    X, y = create_sliding_windows(kpi_data, labels, window_size=data_window)
    X_tr, y_tr, X_val, y_val, _, _ = train_val_test_split(X, y)

    normalizer = KPINormalizer()
    X_tr  = normalizer.fit_transform(X_tr)
    X_val = normalizer.transform(X_val)

    device = torch.device("cpu")
    prior  = torch.tensor(adj, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr,  dtype=torch.float32),
                      torch.tensor(y_tr,  dtype=torch.long)),
        batch_size=32, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                      torch.tensor(y_val, dtype=torch.long)),
        batch_size=32, shuffle=False, num_workers=0,
    )

    # ── Stage 2: model + optimiser setup ─────────────────────────────────────
    n_cells = int(kpi_data.shape[1])
    n_kpis  = int(kpi_data.shape[2])

    model = Simba(n_kpis=n_kpis, n_cells=n_cells, window_size=data_window).to(device)

    class_weights = compute_class_weights(y_tr)
    criterion = WeightedFocalLoss(
        n_classes=Simba.N_CLASSES,
        class_weights=class_weights,
        gamma=2.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # ── Stage 3: training loop with milestone SSE events ─────────────────────
    # Milestones: (epoch_threshold, progress_pct, message)
    milestones = [
        (max(1,     epochs // 3), 40, "Learning traffic patterns..."),
        (max(2, 2 * epochs // 3), 65, "Learning signal quality baselines..."),
        (max(3, 5 * epochs // 6), 85, "Learning capacity thresholds..."),
    ]
    sent = set()

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_loader, optimizer, criterion, device, prior)

        for threshold, progress, message in milestones:
            if epoch >= threshold and progress not in sent:
                emit({"stage": "training", "message": message, "progress": progress})
                sent.add(progress)

    # ── Stage 4: evaluate and persist ────────────────────────────────────────
    _, val_metrics = evaluate(model, val_loader, criterion, device, prior)

    ms.store(
        model=model,
        normalizer=normalizer,
        prior=prior,
        adjacency=adj,
        config={
            "epochs":          epochs,
            "data_window":     data_window,
            "n_cells":         n_cells,
            "n_kpis":          n_kpis,
            "val_macro_f1":    round(float(val_metrics["macro_f1"]),    4),
            "val_anomaly_f1":  round(float(val_metrics["anomaly_f1"]), 4),
        },
    )

    emit({"stage": "complete",
          "message": "AI engine armed — monitoring 47 cells",
          "progress": 100})


@router.post("/train_model")
async def train_model(body: TrainModelRequest) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _run() -> None:
        try:
            _do_training(body.epochs, body.data_window, _emit)
        except Exception as exc:
            _emit({"stage": "error", "message": str(exc), "progress": -1})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    threading.Thread(target=_run, daemon=True).start()

    async def event_stream():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",  # prevent nginx from buffering SSE
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /tools/correlate_alarms
# ─────────────────────────────────────────────────────────────────────────────

class CorrelateAlarmsRequest(BaseModel):
    alarm_ids: list[str] = []       # empty → all currently raised alarms
    include_cleared: bool = False   # include alarms with state='cleared'


def _load_alarms(session, alarm_ids: list[str], include_cleared: bool) -> dict:
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
        alarm["triggered_on"]    = r["triggered_on"]
        alarm["affected_services"] = r["services"]
        result[aid] = alarm
    return result


def _propagation_groups(
    session, scope_ids: set[str], alarms_by_id: dict
) -> tuple[list[dict], set[str]]:
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
    covered: set[str]  = set()

    for row in rows:
        root_id     = row["root_id"]
        symptom_ids = [s for s in (row["symptom_ids"] or []) if s]
        all_ids     = [root_id] + symptom_ids
        covered.update(all_ids)

        affected: set[str] = set()
        for aid in all_ids:
            affected.update(alarms_by_id.get(aid, {}).get("affected_services", []))

        groups.append({
            "group_id":        f"PROP-{root_id}",
            "correlation_type": "propagation",
            "root_cause_id":   root_id,
            "root_node_id":    row["root_node_id"],
            "alarm_count":     len(all_ids),
            "alarms":          [alarms_by_id[a] for a in all_ids if a in alarms_by_id],
            "affected_services": sorted(affected),
        })

    return groups, covered


def _service_groups(
    session, remaining_ids: set[str], alarms_by_id: dict
) -> tuple[list[dict], set[str]]:
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
    covered: set[str]  = set()

    for row in rows:
        aids = row["alarm_ids"]
        covered.update(aids)
        groups.append({
            "group_id":        f"SVC-{row['service_id']}",
            "correlation_type": "service",
            "root_cause_id":   None,
            "root_node_id":    None,
            "alarm_count":     len(aids),
            "alarms":          [alarms_by_id[a] for a in aids if a in alarms_by_id],
            "affected_services": [row["service_id"]],
        })

    return groups, covered


def _isolated_groups(isolated_ids: set[str], alarms_by_id: dict) -> list[dict]:
    return [
        {
            "group_id":        f"ISO-{aid}",
            "correlation_type": "isolated",
            "root_cause_id":   None,
            "root_node_id":    None,
            "alarm_count":     1,
            "alarms":          [alarms_by_id[aid]],
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
        scope_ids    = set(alarms_by_id)

        prop_groups, prop_ids = _propagation_groups(session, scope_ids, alarms_by_id)
        remaining             = scope_ids - prop_ids
        svc_groups, svc_ids   = _service_groups(session, remaining, alarms_by_id)
        isolated              = _isolated_groups(remaining - svc_ids, alarms_by_id)

    groups = prop_groups + svc_groups + isolated

    return {
        "status":      "ok",
        "tool":        "correlate_alarms",
        "alarm_count": len(scope_ids),
        "group_count": len(groups),
        "groups":      groups,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stubs
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/run_inference")
def run_inference() -> dict:
    return {"status": "ok", "tool": "run_inference"}


@router.post("/get_rca")
def get_rca() -> dict:
    return {"status": "ok", "tool": "get_rca"}


@router.post("/ask_assistant")
def ask_assistant() -> dict:
    return {"status": "ok", "tool": "ask_assistant"}
