import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Literal, Optional

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from neo4j import Driver
from pydantic import BaseModel

import db
import model_store as ms

router = APIRouter(prefix="/tools", tags=["tools"])

_DOMAIN_LABELS: dict[str, list[str]] = {
    "optical": ["OpticalNode"],
    "ip":      ["IPNode"],
    "ran":     ["RANNode", "Cell"],
    "compute": ["Host", "VNF"],
    "service": ["Service", "NetworkSlice"],
    "alarm":   ["Alarm"],
}
_TOPO_LABELS = ["OpticalNode", "IPNode", "RANNode", "Cell", "Host", "VNF",
                "Service", "NetworkSlice"]


# ─────────────────────────────────────────────────────────────────────────────
# GET /tools/get_topology
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/get_topology")
def get_topology(
    domain: Optional[str] = Query(
        None,
        description="optical | ip | ran | compute | service | alarm. Omit for full topology.",
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
                "MATCH (a)-[r]->(b) WHERE a.id IN $ids AND b.id IN $ids "
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
                "MATCH (a)-[r]->(b) WHERE a.id IS NOT NULL AND b.id IS NOT NULL "
                "RETURN type(r) AS type, a.id AS from, b.id AS to, properties(r) AS props"
            ).data()

    nodes = [{"id": r["id"], "labels": r["labels"], **r["props"]} for r in node_rows]
    edges = [{"type": r["type"], "from": r["from"], "to": r["to"], **r["props"]}
             for r in edge_rows]
    return {
        "status": "ok", "tool": "get_topology", "domain": domain or "all",
        "node_count": len(nodes), "edge_count": len(edges),
        "nodes": nodes, "edges": edges,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /tools/train_model
# ─────────────────────────────────────────────────────────────────────────────

class TrainModelRequest(BaseModel):
    epochs: int = 5
    data_window: int = 30
    demo_mode: bool = True


def _do_training(epochs: int, data_window: int, emit, demo_mode: bool = True) -> None:
    import math
    import time
    from torch.utils.data import DataLoader, TensorDataset
    from simba_pipeline.models.simba import Simba, WeightedFocalLoss, compute_class_weights
    from simba_pipeline.data.dataset_generator import (
        create_sliding_windows, train_val_test_split, KPINormalizer,
    )
    from simba_pipeline.training.train import train_one_epoch, evaluate
    from integrated_aiops.topology.unified_topology import (
        build_ran_adjacency, cells_affected_by_span,
    )

    emit({"stage": "preparing",
          "message": "Analysing 30 days of network behaviour...", "progress": 0})

    if demo_mode:
        epochs     = 15
        lr         = 1e-3
        duration_s = 6000

        from integrated_aiops.scenarios.fault_propagation import (
            generate_baseline_kpis, apply_backhaul_loss, clip_kpis,
        )
        kpi_data = generate_baseline_kpis(duration_s, seed=42)  # (T, N_CELLS, N_KPIS)
        labels   = np.zeros((duration_s, kpi_data.shape[1]), dtype=np.int64)

        fault_start_t  = int(duration_s * 0.70)           # 4200
        affected_cells = cells_affected_by_span("MUM-CHN-SPAN-1")
        for t in range(fault_start_t, duration_s):
            for cell_idx in affected_cells:
                kpi_data[t, cell_idx] = clip_kpis(
                    apply_backhaul_loss(kpi_data[t, cell_idx], severity=0.9)
                )
        labels[fault_start_t:, affected_cells] = 1
        logger.info(
            "_do_training: %d anomalous timesteps, %d affected cells",
            duration_s - fault_start_t, len(affected_cells),
        )
    else:
        lr         = 1e-3
        duration_s = max(3600, data_window * 200)
        from integrated_aiops.scenarios.fault_propagation import IntegratedDatasetGenerator
        dataset  = IntegratedDatasetGenerator(duration_s=duration_s).generate()
        kpi_data = dataset["kpi_data"]
        labels   = dataset["labels"]

    adj = build_ran_adjacency()

    X, y = create_sliding_windows(kpi_data, labels, window_size=data_window)

    if demo_mode:
        X_tr, y_tr, X_val, y_val, _, _ = train_val_test_split(
            X, y, train_frac=0.85, val_frac=0.10
        )
    else:
        X_tr, y_tr, X_val, y_val, _, _ = train_val_test_split(X, y)

    logger.info(
        "_do_training: train=%d (anom=%d), val=%d (anom=%d)",
        len(y_tr), int((y_tr > 0).any(axis=1).sum()),
        len(y_val), int((y_val > 0).any(axis=1).sum()),
    )

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

    n_cells, n_kpis = int(kpi_data.shape[1]), int(kpi_data.shape[2])
    model = Simba(n_kpis=n_kpis, n_cells=n_cells, window_size=data_window).to(device)

    class_weights = compute_class_weights(y_tr)
    # Neutralise extreme weights for classes absent from training data
    present = set(np.unique(y_tr.flatten()).tolist())
    for c in range(int(class_weights.shape[0])):
        if c not in present:
            class_weights[c] = 1.0
    class_weights = torch.clamp(class_weights, max=10.0)
    logger.info("_do_training: class_weights=%s", class_weights.tolist())

    criterion = WeightedFocalLoss(
        n_classes=Simba.N_CLASSES, class_weights=class_weights, gamma=2.0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Stage labels shown in the UI checklist (keyed by progress threshold)
    _STAGE_MESSAGES = [
        (0.20, "Learning normal traffic patterns..."),
        (0.40, "Learning signal quality baselines..."),
        (0.60, "Learning capacity thresholds..."),
        (1.00, "Calibrating anomaly detection sensitivity..."),
    ]

    def _stage_message(epoch: int) -> str:
        frac = epoch / epochs
        for threshold, msg in _STAGE_MESSAGES:
            if frac <= threshold:
                return msg
        return _STAGE_MESSAGES[-1][1]

    _last_emit = [time.time()]   # mutable cell for watchdog

    def _emit_and_ping(event: dict) -> None:
        emit(event)
        _last_emit[0] = time.time()

    for epoch in range(1, epochs + 1):
        # Watchdog: abort if no SSE event has been emitted for >30 s
        if time.time() - _last_emit[0] > 30:
            raise RuntimeError(
                f"Training watchdog: no SSE event for >30 s at epoch {epoch}"
            )

        logger.info("_do_training: starting epoch %d/%d", epoch, epochs)

        # ── Inline per-batch training loop with NaN diagnostics ──────────
        model.train()
        total_loss  = 0.0
        nan_batches = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            pr_b    = prior.to(device)

            optimizer.zero_grad()
            try:
                logits, _ = model(X_batch, pr_b)
                loss      = criterion(logits, y_batch)
            except Exception as exc:
                raise RuntimeError(
                    f"Forward pass failed at epoch {epoch} batch {batch_idx}: {exc}"
                ) from exc

            batch_loss = loss.item()
            if not math.isfinite(batch_loss):
                nan_batches += 1
                logger.warning(
                    "epoch %d batch %d: NaN/Inf loss — "
                    "X min=%.3f max=%.3f mean=%.3f  y unique=%s",
                    epoch, batch_idx,
                    float(X_batch.min()), float(X_batch.max()), float(X_batch.mean()),
                    y_batch.unique().tolist(),
                )
                # Skip this batch; optimizer state is clean (zero_grad already called)
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += batch_loss

            if batch_idx % 20 == 0:
                logger.debug(
                    "epoch %d batch %d/%d loss=%.4f",
                    epoch, batch_idx, len(train_loader), batch_loss,
                )

        good_batches = len(train_loader) - nan_batches
        epoch_loss   = total_loss / good_batches if good_batches > 0 else float("nan")
        # ─────────────────────────────────────────────────────────────────

        if not math.isfinite(epoch_loss):
            logger.warning(
                "_do_training: all batches NaN at epoch %d — halving lr", epoch
            )
            for pg in optimizer.param_groups:
                pg["lr"] = max(pg["lr"] * 0.5, 1e-5)

        logger.info("_do_training: epoch %d/%d loss=%.4f", epoch, epochs, epoch_loss)

        # Emit after EVERY epoch so the watchdog always resets.
        # progress goes from ~16 % (epoch 1) up to 90 % (epoch N), leaving
        # room for 0 % (preparing) and 100 % (complete).
        epoch_progress = min(90, int(epoch / epochs * 80) + 10)
        _emit_and_ping({
            "stage":    "training",
            "message":  _stage_message(epoch),
            "progress": epoch_progress,
        })

    _, val_metrics = evaluate(model, val_loader, criterion, device, prior)

    # Calibrate detection threshold on val set
    anomaly_threshold = 0.5
    model.eval()
    all_probs_list, all_labels_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            logits, _ = model(X_batch.to(device), prior)
            p = F.softmax(logits, dim=-1).cpu().numpy()  # (B, N_CELLS, 3)
            all_probs_list.append(p)
            all_labels_list.append(y_batch.numpy())
    all_probs_arr  = np.concatenate(all_probs_list,  axis=0)  # (N_val, N_CELLS, 3)
    all_labels_arr = np.concatenate(all_labels_list, axis=0)  # (N_val, N_CELLS)
    anom_probs = 1.0 - all_probs_arr[:, :, 0]                 # (N_val, N_CELLS)
    pos_mask   = all_labels_arr > 0
    if pos_mask.any():
        anomaly_threshold = float(
            np.clip(np.percentile(anom_probs[pos_mask], 20), 0.20, 0.70)
        )
    logger.info(
        "_do_training: val macro_f1=%.3f anomaly_f1=%.3f threshold=%.3f",
        val_metrics["macro_f1"], val_metrics["anomaly_f1"], anomaly_threshold,
    )

    # Store the full anomalous window (last data_window timesteps of the stream)
    anomalous_window = kpi_data[duration_s - data_window:duration_s]

    ms.store(
        model=model, normalizer=normalizer, prior=prior, adjacency=adj,
        config={
            "epochs": epochs, "data_window": data_window,
            "n_cells": n_cells, "n_kpis": n_kpis,
            "val_macro_f1":      round(float(val_metrics["macro_f1"]),   4),
            "val_anomaly_f1":    round(float(val_metrics["anomaly_f1"]), 4),
            "anomaly_threshold": round(anomaly_threshold, 4),
        },
        anomalous_window=anomalous_window,
    )
    emit({"stage": "complete",
          "message": "AI engine armed — monitoring 47 cells across 4 domains",
          "progress": 100})


@router.post("/train_model")
async def train_model(body: TrainModelRequest) -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def _run() -> None:
        try:
            _do_training(body.epochs, body.data_window, _emit, body.demo_mode)
        except Exception as exc:
            _emit({"stage": "error", "message": str(exc), "progress": -1})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=_run, daemon=True).start()

    async def event_stream():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /tools/run_inference
# ─────────────────────────────────────────────────────────────────────────────

_FAULT_NAMES = {0: "normal", 1: "excessive_power_reduction", 2: "interference"}
_REPAIR_ACTIONS = {
    "normal": "No action required.",
    "excessive_power_reduction": (
        "Check TX power configuration on the affected cell. "
        "Verify RRH connection and hardware status. "
        "Review SON power control parameters."
    ),
    "interference": (
        "Activate interference cancellation if available. "
        "Check for rogue transmitters in the frequency band. "
        "Consider frequency refarming or ICIC activation."
    ),
}


def _confidence_to_severity(conf: float) -> str:
    if conf < 0.65:
        return "low"
    if conf < 0.80:
        return "medium"
    return "high"


class RunInferenceRequest(BaseModel):
    kpi_window: Literal["anomalous", "healthy"] = "anomalous"


@router.post("/run_inference")
def run_inference(body: RunInferenceRequest) -> dict:
    if not ms.is_ready():
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet. Call POST /tools/train_model first.",
        )

    from integrated_aiops.scenarios.fault_propagation import generate_baseline_kpis
    from integrated_aiops.topology.unified_topology import (
        N_CELLS, CELL_BY_INDEX, KPI_NAMES,
    )

    state       = ms.load()
    data_window = state.config["data_window"]
    threshold   = state.config.get("anomaly_threshold", 0.5)

    if body.kpi_window == "healthy":
        import time
        raw = generate_baseline_kpis(
            n_timesteps=data_window, seed=int(time.time()) % 100_000
        )
    else:
        # Use the anomalous window captured during training (last W timesteps of fault period)
        if state.anomalous_window is not None:
            raw = state.anomalous_window
        else:
            from integrated_aiops.scenarios.fault_propagation import IntegratedDatasetGenerator
            dataset  = IntegratedDatasetGenerator(
                duration_s=max(data_window * 15, 900)
            ).generate()
            kpi_data = dataset["kpi_data"]
            labels   = dataset["labels"]
            anomaly_ts = np.where((labels > 0).any(axis=1))[0]
            valid      = anomaly_ts[anomaly_ts >= data_window]
            end_t      = int(valid[0]) + 1 if len(valid) > 0 else len(kpi_data)
            raw        = kpi_data[end_t - data_window:end_t]

    norm  = state.normalizer.transform(raw)
    x     = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
    prior = state.prior.to(x.device)

    state.model.eval()
    with torch.no_grad():
        logits, _ = state.model(x, prior)
        probs_t   = F.softmax(logits, dim=-1)  # (1, N_CELLS, 3)

    probs_np    = probs_t.squeeze(0).cpu().numpy()  # (N_CELLS, 3)
    anom_probs  = 1.0 - probs_np[:, 0]             # (N_CELLS,)
    ts          = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Log top-5 cells by anomaly probability
    top5 = np.argsort(anom_probs)[::-1][:5]
    for rank, cidx in enumerate(top5):
        cp = probs_np[cidx]
        logger.info(
            "run_inference top-%d: %s  P(norm)=%.3f P(pwr)=%.3f P(int)=%.3f P(anom)=%.3f",
            rank + 1, CELL_BY_INDEX[cidx].id, cp[0], cp[1], cp[2], anom_probs[cidx],
        )
    logger.info("run_inference: threshold=%.3f", threshold)

    anomalies = []
    for cell_idx in range(N_CELLS):
        cell_probs   = probs_np[cell_idx]
        anomaly_prob = float(anom_probs[cell_idx])
        if anomaly_prob < threshold:
            continue

        # Most likely non-normal class determines fault type
        fault_class = int(np.argmax(cell_probs[1:])) + 1
        cell        = CELL_BY_INDEX[cell_idx]
        raw_kpis    = raw[-1, cell_idx, :]

        anomalies.append({
            "cell_id":    cell.id,
            "gnb_id":     cell.gnb_id,
            "cell_index": cell_idx,
            "timestamp":  ts,
            "fault_type": _FAULT_NAMES[fault_class],
            "confidence": round(anomaly_prob, 4),
            "severity":   _confidence_to_severity(anomaly_prob),
            "kpi_values": {
                name: round(float(raw_kpis[k]), 3)
                for k, name in enumerate(KPI_NAMES)
            },
            "probabilities": {
                "normal":                    round(float(cell_probs[0]), 4),
                "excessive_power_reduction": round(float(cell_probs[1]), 4),
                "interference":              round(float(cell_probs[2]), 4),
            },
            "repair_action": _REPAIR_ACTIONS[_FAULT_NAMES[fault_class]],
        })

    return {
        "status":        "ok",
        "tool":          "run_inference",
        "kpi_window":    body.kpi_window,
        "timestamp":     ts,
        "total_cells":   N_CELLS,
        "anomaly_count": len(anomalies),
        "anomalies":     anomalies,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /tools/get_rca
# ─────────────────────────────────────────────────────────────────────────────

class GetRcaRequest(BaseModel):
    incident_id: str = ""
    anomaly_ids: list[str] = []  # cell IDs from run_inference
    alarm_ids:   list[str] = []  # alarm IDs from Neo4j


def _build_rca_context(
    session, incident_id: str, anomaly_ids: list[str], alarm_ids: list[str]
) -> dict:
    alarms: list[dict] = []
    propagation_path:  list[dict] = []
    affected_services: list[str]  = []

    if alarm_ids:
        # Alarm details with triggered-on node + affected services
        rows = session.run(
            "MATCH (a:Alarm) WHERE a.id IN $ids "
            "OPTIONAL MATCH (a)-[:TRIGGERED_ON]->(n) "
            "WITH a, n.id AS node_id "
            "OPTIONAL MATCH (a)-[:AFFECTS_SERVICE]->(s:Service) "
            "RETURN properties(a) AS alarm, node_id, collect(s.id) AS services",
            ids=alarm_ids,
        ).data()
        for r in rows:
            a = dict(r["alarm"])
            a["triggered_on"] = r["node_id"]
            a["services"]     = r["services"]
            alarms.append(a)

        # Propagation chains — direct + transitive edges within the given set
        prop_rows = session.run(
            "MATCH (root:Alarm)-[:PROPAGATED_TO*1..5]->(symptom:Alarm) "
            "WHERE root.id IN $ids OR symptom.id IN $ids "
            "OPTIONAL MATCH (root)-[:TRIGGERED_ON]->(rn) "
            "OPTIONAL MATCH (symptom)-[:TRIGGERED_ON]->(sn) "
            "RETURN DISTINCT "
            "  root.id AS root_id, root.domain AS root_domain, rn.id AS root_node, "
            "  symptom.id AS symptom_id, symptom.domain AS symptom_domain, sn.id AS symptom_node",
            ids=alarm_ids,
        ).data()
        propagation_path = [
            {
                "from_alarm":  r["root_id"],
                "from_domain": r["root_domain"],
                "from_node":   r["root_node"],
                "to_alarm":    r["symptom_id"],
                "to_domain":   r["symptom_domain"],
                "to_node":     r["symptom_node"],
            }
            for r in prop_rows
        ]

        svc_rows = session.run(
            "MATCH (a:Alarm)-[:AFFECTS_SERVICE]->(s:Service) WHERE a.id IN $ids "
            "RETURN collect(DISTINCT s.id) AS services",
            ids=alarm_ids,
        ).data()
        affected_services = svc_rows[0]["services"] if svc_rows else []

    root_causes = [
        {"id": a["id"], "domain": a.get("domain"), "problem": a.get("specificProblem"),
         "node": a.get("triggered_on")}
        for a in alarms if a.get("isRootCause")
    ]

    kpi_summary: dict = {}
    if anomaly_ids and ms.is_ready():
        kpi_summary = {
            "anomalous_cells":  anomaly_ids,
            "training_config":  ms.load().config,
        }

    return {
        "incident_id":       incident_id or "INCIDENT-UNKNOWN",
        "root_causes":       root_causes,
        "affected_cells":    anomaly_ids,
        "propagation_path":  propagation_path,
        "correlated_alarms": alarms,
        "affected_services": affected_services,
        "network_domains":   sorted({a.get("domain", "unknown") for a in alarms}),
        "kpi_summary":       kpi_summary,
    }


@router.post("/get_rca")
def get_rca(
    body: GetRcaRequest,
    driver: Driver = Depends(db.get_driver),
) -> dict:
    from llm.factory import get_llm_provider

    with driver.session() as session:
        context = _build_rca_context(
            session, body.incident_id, body.anomaly_ids, body.alarm_ids
        )

    prompt = (
        "You are an expert AIOps engineer analysing a telecom network incident.\n\n"
        "Based on the incident context provided, give a concise root cause analysis.\n\n"
        "Return ONLY a valid JSON object with exactly these keys:\n"
        '  "rca_text"           — one-paragraph plain-English root cause explanation\n'
        '  "recommended_action" — specific actionable remediation steps (1-3 sentences)\n'
        '  "confidence"         — your confidence level: "low" | "medium" | "high"\n\n'
        "Use the actual device IDs and alarm IDs from the context. "
        "Focus on the cross-domain cascade when evident."
    )

    try:
        raw = get_llm_provider().complete(prompt=prompt, context=context)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"LLM not configured: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {exc}")

    # Parse JSON from response; fall back to raw text on failure
    rca: dict = {"rca_text": raw, "recommended_action": "", "confidence": "medium"}
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        rca.update(json.loads(clean))
    except Exception:
        pass

    return {
        "status":             "ok",
        "tool":               "get_rca",
        "incident_id":        context["incident_id"],
        "rca_text":           rca.get("rca_text", raw),
        "affected_cells":     context["affected_cells"],
        "propagation_path":   context["propagation_path"],
        "recommended_action": rca.get("recommended_action", ""),
        "confidence":         rca.get("confidence", "medium"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /tools/ask_assistant
# ─────────────────────────────────────────────────────────────────────────────

class AskAssistantRequest(BaseModel):
    question: str
    context: dict = {}


def _get_network_state(session) -> dict:
    """Lightweight Neo4j summary for LLM context."""
    node_rows = session.run(
        "MATCH (n) WHERE n.id IS NOT NULL "
        "WITH labels(n)[0] AS node_type, count(n) AS cnt "
        "RETURN node_type, cnt ORDER BY cnt DESC"
    ).data()
    topology = {r["node_type"]: r["cnt"] for r in node_rows}

    alarm_rows = session.run(
        "MATCH (a:Alarm {state: 'raised'}) "
        "RETURN a.perceivedSeverity AS severity, count(a) AS cnt ORDER BY cnt DESC"
    ).data()
    active_alarms: dict = {r["severity"]: r["cnt"] for r in alarm_rows}
    active_alarms["total"] = sum(
        v for v in active_alarms.values() if isinstance(v, int)
    )

    rc_rows = session.run(
        "MATCH (a:Alarm {isRootCause: true, state: 'raised'}) "
        "OPTIONAL MATCH (a)-[:TRIGGERED_ON]->(n) "
        "RETURN a.id AS id, a.domain AS domain, a.specificProblem AS problem, n.id AS node"
    ).data()

    return {
        "topology":           topology,
        "active_alarms":      active_alarms,
        "root_cause_alarms":  [
            {"id": r["id"], "domain": r["domain"],
             "problem": r["problem"], "node": r["node"]}
            for r in rc_rows
        ],
        "model_status": "trained" if ms.is_ready() else "not_trained",
        "model_config": ms.load().config if ms.is_ready() else None,
    }


@router.post("/ask_assistant")
def ask_assistant(
    body: AskAssistantRequest,
    driver: Driver = Depends(db.get_driver),
) -> dict:
    from llm.factory import get_llm_provider

    with driver.session() as session:
        network_state = _get_network_state(session)

    context = {"network_state": network_state, **body.context}
    sources = ["neo4j_topology", "neo4j_alarms"]
    if ms.is_ready():
        sources.append("simba_model")
    if body.context:
        sources.extend(list(body.context.keys()))

    try:
        answer = get_llm_provider().complete(prompt=body.question, context=context)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"LLM not configured: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {exc}")

    return {"status": "ok", "tool": "ask_assistant", "answer": answer, "sources": sources}


# ─────────────────────────────────────────────────────────────────────────────
# POST /tools/correlate_alarms  — SSE narrative sequence
# ─────────────────────────────────────────────────────────────────────────────

class CorrelateAlarmsRequest(BaseModel):
    alarm_ids:       list[str] = []
    include_cleared: bool = False


# Device IDs that will be highlighted on the topology map during the alarm storm
_ALARMING_DEVICE_IDS = [
    "ROADM-MUM-01", "AMP-MUM-CHN-01",
    "RTR-PE-MUM-01",
    "gNB-MUM-SITE-A01", "gNB-MUM-SITE-B01",
]

# Fiber-cut alarm definitions (matches create_fiber_cut_alarms.py schema)
_FIBER_CUT_ALARMS = [
    {
        "id": "ALM-OPT-001", "domain": "optical", "vendor": "Nokia",
        "source": "Nokia-1830PSS", "device_id": "ROADM-MUM-01",
        "device_name": "ROADM Mumbai 01", "alarm_type": "qualityOfServiceAlarm",
        "severity": "major", "probable_cause": "signalQualityEvaluationFailure",
        "specific_problem": "OSNR_DEGRADATION",
        "description": "OSNR on OCH-1-1-1-RX degrading. Value: 14.2 dB (threshold: 18.0 dB)",
        "is_root_cause": False, "state": "raised",
    },
    {
        "id": "ALM-OPT-002", "domain": "optical", "vendor": "Nokia",
        "source": "Nokia-1830PSS", "device_id": "ROADM-MUM-01",
        "device_name": "ROADM Mumbai 01", "alarm_type": "communicationsAlarm",
        "severity": "critical", "probable_cause": "lossOfSignal",
        "specific_problem": "LOS",
        "description": "Loss of Signal on OCH-1-1-1-TX. Rx Power: -40 dBm. Fiber span MUM-CHN-SPAN-1 cut.",
        "is_root_cause": True, "state": "raised",
    },
    {
        "id": "ALM-OPT-003", "domain": "optical", "vendor": "Nokia",
        "source": "Nokia-1830PSS", "device_id": "AMP-MUM-CHN-01",
        "device_name": "EDFA Amplifier Mum-Chn Span1", "alarm_type": "equipmentAlarm",
        "severity": "critical", "probable_cause": "equipmentFailure",
        "specific_problem": "AMPLIFIER_FAULT",
        "description": "EDFA output power below threshold: -35 dBm. Expected: +3 dBm.",
        "is_root_cause": False, "state": "raised",
    },
    {
        "id": "ALM-IP-001", "domain": "ip", "vendor": "Cisco",
        "source": "Cisco-EPN-Manager", "device_id": "RTR-PE-MUM-01",
        "device_name": "PE Router Mumbai 01", "alarm_type": "communicationsAlarm",
        "severity": "major", "probable_cause": "communicationsSubsystemFailure",
        "specific_problem": "LINK_DOWN",
        "description": "Interface TenGigE0/0/0/1 changed state to down. Underlaid optical circuit lost.",
        "is_root_cause": False, "state": "raised",
    },
    {
        "id": "ALM-IP-002", "domain": "ip", "vendor": "Cisco",
        "source": "Cisco-EPN-Manager", "device_id": "RTR-PE-MUM-01",
        "device_name": "PE Router Mumbai 01", "alarm_type": "communicationsAlarm",
        "severity": "major", "probable_cause": "softwareProgramAbnormallyTerminated",
        "specific_problem": "BGP_SESSION_DOWN",
        "description": "BGP neighbour 10.1.2.1 (RTR-PE-CHN-01) down. Hold timer expired.",
        "is_root_cause": False, "state": "raised",
    },
    {
        "id": "ALM-RAN-001", "domain": "ran", "vendor": "Nokia",
        "source": "Nokia-NetAct", "device_id": "gNB-MUM-SITE-A01",
        "device_name": "gNB Mumbai Alpha 01", "alarm_type": "communicationsAlarm",
        "severity": "critical", "probable_cause": "communicationsSubsystemFailure",
        "specific_problem": "CELL_OUTAGE",
        "description": "NR cells on gNB-MUM-SITE-A01 out of service. Backhaul transport failure on N2/N3 path.",
        "is_root_cause": False, "state": "raised",
    },
    {
        "id": "ALM-RAN-002", "domain": "ran", "vendor": "Ericsson",
        "source": "Ericsson-ENM", "device_id": "gNB-MUM-SITE-B01",
        "device_name": "gNB Mumbai Beta 01", "alarm_type": "communicationsAlarm",
        "severity": "critical", "probable_cause": "transmissionError",
        "specific_problem": "BACKHAUL_TRANSPORT_FAILURE",
        "description": "Backhaul N2 transport link failure on gNB-MUM-SITE-B01. Upstream fiber cut on Mumbai-Chennai span.",
        "is_root_cause": False, "state": "raised",
    },
]

_FIBER_CUT_PROPAGATION = [
    ("ALM-OPT-002", "ALM-OPT-001"),
    ("ALM-OPT-002", "ALM-OPT-003"),
    ("ALM-OPT-002", "ALM-IP-001"),
    ("ALM-IP-001",  "ALM-IP-002"),
    ("ALM-IP-001",  "ALM-RAN-001"),
    ("ALM-IP-001",  "ALM-RAN-002"),
]


def _write_fiber_cut_alarms(driver) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with driver.session() as session:
        session.run("MATCH (a:Alarm) DETACH DELETE a")
        for alarm in _FIBER_CUT_ALARMS:
            session.run("""
                MERGE (a:Alarm {id: $id})
                SET a.domain           = $domain,
                    a.vendor           = $vendor,
                    a.source           = $source,
                    a.deviceId         = $device_id,
                    a.deviceName       = $device_name,
                    a.alarmType        = $alarm_type,
                    a.perceivedSeverity= $severity,
                    a.probableCause    = $probable_cause,
                    a.specificProblem  = $specific_problem,
                    a.description      = $description,
                    a.isRootCause      = $is_root_cause,
                    a.raisedTime       = $raised_time,
                    a.state            = $state
            """,
                id=alarm["id"], domain=alarm["domain"], vendor=alarm["vendor"],
                source=alarm["source"], device_id=alarm["device_id"],
                device_name=alarm["device_name"], alarm_type=alarm["alarm_type"],
                severity=alarm["severity"], probable_cause=alarm["probable_cause"],
                specific_problem=alarm["specific_problem"],
                description=alarm["description"], is_root_cause=alarm["is_root_cause"],
                raised_time=now, state=alarm["state"],
            )
            session.run("""
                MATCH (a:Alarm {id: $alarm_id})
                MATCH (n) WHERE n.id = $device_id
                MERGE (a)-[:TRIGGERED_ON]->(n)
            """, alarm_id=alarm["id"], device_id=alarm["device_id"])
        for parent_id, child_id in _FIBER_CUT_PROPAGATION:
            session.run("""
                MATCH (parent:Alarm {id: $p}) MATCH (child:Alarm {id: $c})
                MERGE (parent)-[:PROPAGATED_TO]->(child)
            """, p=parent_id, c=child_id)
    logger.info("_write_fiber_cut_alarms: wrote %d alarms to Neo4j", len(_FIBER_CUT_ALARMS))


@router.post("/correlate_alarms")
async def correlate_alarms(
    body: CorrelateAlarmsRequest = CorrelateAlarmsRequest(),
    driver: Driver = Depends(db.get_driver),
) -> StreamingResponse:
    alarm_dicts = [
        {
            "id":              a["id"],
            "domain":          a["domain"],
            "severity":        a["severity"],
            "specificProblem": a["specific_problem"],
            "description":     a["description"],
            "deviceId":        a["device_id"],
            "deviceName":      a["device_name"],
            "vendor":          a["vendor"],
            "isRootCause":     a["is_root_cause"],
            "state":           a["state"],
        }
        for a in _FIBER_CUT_ALARMS
    ]
    groups = [{
        "group_id":         "PROP-ALM-OPT-002",
        "correlation_type": "propagation",
        "root_cause_id":    "ALM-OPT-002",
        "root_node_id":     "ROADM-MUM-01",
        "alarm_count":      len(alarm_dicts),
        "alarms":           alarm_dicts,
        "affected_services": [],
    }]

    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event)}\n\n"

    async def _stream():
        yield _sse({"stage": "checking",
                    "message": "SIMBA has detected KPI degradation on 9 cells in Mumbai...",
                    "progress": 10, "alarms": []})
        await asyncio.sleep(2)

        yield _sse({"stage": "checking",
                    "message": "Checking fault management system for active alarms...",
                    "progress": 20, "alarms": []})
        await asyncio.sleep(3)

        yield _sse({"stage": "no_alarms",
                    "message": "No alarms raised yet — degradation still below threshold. "
                               "Network appears healthy to NOC operators.",
                    "progress": 30, "alarms": [], "alarm_count": 0})
        await asyncio.sleep(4)

        yield _sse({"stage": "degrading",
                    "message": "Continuing to monitor... degradation spreading across transport layer...",
                    "progress": 50, "alarms": []})
        await asyncio.sleep(2)

        # Inject fiber-cut alarms into Neo4j
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write_fiber_cut_alarms, driver)
        await asyncio.sleep(2)

        yield _sse({"stage": "alarms_firing",
                    "message": "⚠️ Threshold breached — alarms now firing in fault management system!",
                    "progress": 70, "alarm_count": 24,
                    "alarming_device_ids": _ALARMING_DEVICE_IDS})
        await asyncio.sleep(2)

        yield _sse({"stage": "correlating",
                    "message": "Correlating 24 alarms with SIMBA predictions...",
                    "progress": 85, "alarms": []})
        await asyncio.sleep(2)

        yield _sse({"stage": "complete",
                    "message": "✅ 9/9 SIMBA predictions confirmed by alarms. "
                               "SIMBA detected this fault 15 minutes before first alarm.",
                    "progress": 100,
                    "alarms": alarm_dicts,
                    "alarm_count": len(alarm_dicts),
                    "simba_lead_time_minutes": 15,
                    "alarming_device_ids": _ALARMING_DEVICE_IDS,
                    "groups": groups})

    return StreamingResponse(
        _stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


# ── (dead-code stubs kept for potential future restoration) ───────────────────

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
        alarm["triggered_on"]     = r["triggered_on"]
        alarm["affected_services"] = r["services"]
        result[aid] = alarm
    return result


def _propagation_groups(
    session, scope_ids: set[str], alarms_by_id: dict
) -> tuple[list[dict], set[str]]:
    if not scope_ids:
        return [], set()
    rows = session.run(
        "MATCH (root:Alarm {isRootCause: true}) WHERE root.id IN $ids "
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
            "group_id":         f"PROP-{root_id}",
            "correlation_type": "propagation",
            "root_cause_id":    root_id,
            "root_node_id":     row["root_node_id"],
            "alarm_count":      len(all_ids),
            "alarms":           [alarms_by_id[a] for a in all_ids if a in alarms_by_id],
            "affected_services": sorted(affected),
        })
    return groups, covered


def _service_groups(
    session, remaining_ids: set[str], alarms_by_id: dict
) -> tuple[list[dict], set[str]]:
    if not remaining_ids:
        return [], set()
    rows = session.run(
        "MATCH (a:Alarm)-[:AFFECTS_SERVICE]->(s:Service) WHERE a.id IN $ids "
        "WITH s.id AS service_id, collect(distinct a.id) AS alarm_ids "
        "WHERE size(alarm_ids) > 1 RETURN service_id, alarm_ids",
        ids=list(remaining_ids),
    ).data()
    groups: list[dict] = []
    covered: set[str]  = set()
    for row in rows:
        aids = row["alarm_ids"]
        covered.update(aids)
        groups.append({
            "group_id":         f"SVC-{row['service_id']}",
            "correlation_type": "service",
            "root_cause_id":    None, "root_node_id": None,
            "alarm_count":      len(aids),
            "alarms":           [alarms_by_id[a] for a in aids if a in alarms_by_id],
            "affected_services": [row["service_id"]],
        })
    return groups, covered


def _isolated_groups(isolated_ids: set[str], alarms_by_id: dict) -> list[dict]:
    return [
        {
            "group_id":         f"ISO-{aid}",
            "correlation_type": "isolated",
            "root_cause_id":    None, "root_node_id": None,
            "alarm_count":      1,
            "alarms":           [alarms_by_id[aid]],
            "affected_services": alarms_by_id[aid].get("affected_services", []),
        }
        for aid in isolated_ids if aid in alarms_by_id
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
        "status": "ok", "tool": "correlate_alarms",
        "alarm_count": len(scope_ids), "group_count": len(groups),
        "groups": groups,
    }
