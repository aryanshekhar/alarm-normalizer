"""
Integrated AIOps Demo
======================
Runs the complete end-to-end pipeline:

  1. Build unified topology (optical + IP + RAN + compute)
  2. Simulate cross-domain fault scenarios (fiber cut → IP → RAN cascade)
  3. Run alarm normalisation pipeline on generated alarm events (TMF642)
  4. Train SIMBA on the KPI stream with ground truth labels
  5. Run SIMBA inference and correlate with normalised alarms
  6. Write everything to Neo4j for visual graph exploration

Usage:
    cd integrated_aiops
    pip install torch numpy neo4j

    # Quick mode (smaller topology, fewer epochs)
    python run_integrated_demo.py --quick

    # Full mode
    python run_integrated_demo.py

    # Skip training (use saved model)
    python run_integrated_demo.py --skip-train
"""

import os
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

from .topology.unified_topology import (
    N_CELLS, ALL_CELLS, GNB_LIST, build_ran_adjacency,
    OPTICAL_NODES, IP_NODES, KPI_NAMES
)
from .scenarios.fault_propagation import (
    IntegratedDatasetGenerator, AlarmEvent
)


def print_banner(title: str):
    print(f"\n{'═'*68}")
    print(f"  {title}")
    print(f"{'═'*68}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Topology summary
# ─────────────────────────────────────────────────────────────────────────────

def print_topology_summary():
    print_banner("Step 1 — Unified Network Topology")
    print(f"  Optical nodes : {len(OPTICAL_NODES)}"
          f" (ROADMs, amplifiers, OTN transponders)")
    print(f"  IP nodes      : {len(IP_NODES)}"
          f" (P-routers, PE-routers, agg switches)")
    print(f"  gNBs          : {len(GNB_LIST)}"
          f" across Mumbai, Chennai, Bangalore, Delhi")
    print(f"  Cells (sectors): {N_CELLS} (3 per gNB)")
    print(f"  KPIs per cell : {len(KPI_NAMES)}")
    print()
    print(f"  Cell → gNB mapping:")
    for gnb in GNB_LIST:
        cell_indices = [str(c.cell_index) for c in gnb.cells]
        print(f"    {gnb.id:28s} [{gnb.vendor:8s}]"
              f"  cells {', '.join(cell_indices)}"
              f"  backhaul→ {gnb.backhaul_pe}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Generate cross-domain fault dataset
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(duration_s: int) -> dict:
    print_banner("Step 2 — Cross-Domain Fault Simulation")
    print(f"  Duration : {duration_s}s ({duration_s//60} minutes)")
    print(f"  Scenarios:")
    print(f"    A. Fiber cut Mumbai-Chennai → IP link loss → RAN cells OOS")
    #print(f"    B. RRH hardware fault on Chennai gNB")
    #print(f"    C. External interference on Bangalore cells")
    print()

    gen     = IntegratedDatasetGenerator(duration_s=duration_s)
    dataset = gen.generate()

    print()
    # Show only first occurrence of each alarm ID for clean demo display
    seen = set()
    unique_alarms = []
    for alarm in dataset['alarm_events']:
        if alarm.alarm_id not in seen:
            seen.add(alarm.alarm_id)
            unique_alarms.append(alarm)
    print(f"  Alarm events generated: {len(unique_alarms)} (fiber cut cascade)")
    for alarm in unique_alarms:
        print(f"    [{alarm.domain:8s}] {alarm.alarm_id:20s}"
              f"  {alarm.perceived_severity:10s}"
              f"  {alarm.specific_problem}")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Alarm normalisation (TMF642)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_alarms(alarm_events: list) -> list:
    """
    Normalise AlarmEvent objects into TMF642-compatible dicts.
    In production this uses the full alarm normalizer pipeline.
    Here we apply the same field mapping logic directly since
    the events are already structured.
    """
    print_banner("Step 3 — Alarm Normalisation (TMF642)")

    import uuid
    normalised = []
    for ev in alarm_events:
        canonical = {
            "@type":              "Alarm",
            "id":                 str(uuid.uuid4()),
            "externalAlarmId":    ev.alarm_id,
            "alarmType":          ev.alarm_type,
            "perceivedSeverity":  ev.perceived_severity,
            "probableCause":      ev.probable_cause,
            "specificProblem":    ev.specific_problem,
            "alarmDetails":       ev.alarm_details,
            "state":              ev.state,
            "alarmRaisedTime":    ev.raised_time,
            "alarmClearedTime":   ev.cleared_time,
            "serviceAffecting":   ev.service_affecting,
            "alarmedObject": {
                "id":   ev.device_id,
                "name": ev.device_name,
            },
            "sourceSystemId":     ev.source_system,
            "x_extensions": {
                "x_domain":      ev.domain,
                "x_vendor":      ev.vendor,
                "x_raw_format":  ev.raw_format,
                "x_scenario_id": ev.scenario_id,
                "x_is_root_cause": ev.is_root_cause,
                "x_propagated_from": ev.propagated_from,
            }
        }
        normalised.append(canonical)
        root_tag = " [ROOT CAUSE]" if ev.is_root_cause else ""
        print(f"  ✓ {ev.alarm_id:22s}  {ev.domain:8s}"
              f"  {ev.perceived_severity:10s}"
              f"  {ev.vendor:10s}{root_tag}")

    print(f"\n  {len(normalised)} alarms normalised to TMF642 canonical format.")
    return normalised


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — SIMBA training
# ─────────────────────────────────────────────────────────────────────────────

def prepare_simba_data(kpi_data: np.ndarray, labels: np.ndarray,
                       window_size: int) -> tuple:
    """Create sliding windows and train/val/test splits."""
    T = kpi_data.shape[0]
    Xs, ys = [], []
    for start in range(0, T - window_size, 1):
        Xs.append(kpi_data[start:start + window_size])
        ys.append(labels[start + window_size - 1])
    X = np.array(Xs, dtype=np.float32)
    y = np.array(ys, dtype=np.int64)

    # Normalise per-KPI
    flat = X.reshape(-1, X.shape[-1])
    mn   = flat.min(axis=0)
    mx   = flat.max(axis=0)
    denom = np.where(mx - mn == 0, 1.0, mx - mn)
    X_norm = (X - mn) / denom

    n = len(X_norm)
    n_tr = int(n * 0.50)
    n_val = int(n * 0.25)
    return (X_norm[:n_tr],       y[:n_tr],
            X_norm[n_tr:n_tr+n_val], y[n_tr:n_tr+n_val],
            X_norm[n_tr+n_val:], y[n_tr+n_val:],
            mn, denom)


def train_simba(dataset: dict, args, model_path: str) -> object:
    print_banner("Step 4 — SIMBA Training on Cross-Domain KPI Stream")

    from simba_pipeline.models.simba import Simba, WeightedFocalLoss, compute_class_weights
    from simba_pipeline.training.train import train, evaluate

    window   = 20 if args.quick else 30
    device   = torch.device("cpu")
    adj_np   = build_ran_adjacency()
    prior    = torch.tensor(adj_np, dtype=torch.float32)

    X_tr, y_tr, X_val, y_val, X_te, y_te, mn, denom = \
        prepare_simba_data(dataset["kpi_data"], dataset["labels"], window)

    print(f"  Window size   : {window}s")
    print(f"  Train windows : {len(X_tr):,}")
    print(f"  Val windows   : {len(X_val):,}")
    print(f"  Test windows  : {len(X_te):,}")
    print(f"  Anomaly rate  : {(y_tr > 0).mean():.3%}")
    print(f"  Fault mapping : 0=normal  1=backhaul/power-reduction  2=interference")

    hidden = 32 if args.quick else 64
    model  = Simba(
        n_kpis=len(KPI_NAMES), n_cells=N_CELLS, window_size=window,
        gcn_hidden=hidden, gcn_output=hidden, temporal_dim=hidden,
        n_heads=4, transformer_layers=1 if args.quick else 2,
        ff_dim=hidden*2, fusion_hidden=hidden*2, dropout=0.1,
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=32, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=32, shuffle=False)

    cw = compute_class_weights(y_tr)
    config = {
        "epochs":       20 if args.quick else 50,
        "batch_size":   32,
        "lr":           1e-3,
        "patience":     10 if args.quick else 20,
        "weight_decay": 1e-4,
        "focal_gamma":  2.0,
        "class_weights": cw,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    train(model, train_loader, val_loader, config, device, prior, model_path)

    # Test evaluation
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_te), torch.tensor(y_te)),
        batch_size=32, shuffle=False)
    criterion = WeightedFocalLoss(class_weights=cw).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, test_metrics = evaluate(model, test_loader, criterion, device, prior)

    print(f"\n  Test Accuracy      : {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro-F1      : {test_metrics['macro_f1']:.4f}")
    print(f"  Test Anomaly-F1    : {test_metrics['anomaly_f1']:.4f}")
    print(f"  Test Anomaly-Recall: {test_metrics['anomaly_recall']:.4f}")

    return model, (mn, denom), adj_np


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — SIMBA inference + alarm correlation
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_and_correlate(model, norm_params, adj_np,
                                dataset: dict, normalised_alarms: list,
                                window: int) -> list:
    print_banner("Step 5 — SIMBA Inference + Alarm Correlation")

    import torch.nn.functional as F
    mn, denom = norm_params
    prior = torch.tensor(adj_np, dtype=torch.float32)

    kpi_data  = dataset["kpi_data"]
    labels    = dataset["labels"]
    scenarios = dataset["scenarios"]
    offsets   = dataset["offsets"]

    FAULT_NAMES = {0:"normal", 1:"excessive_power_reduction", 2:"interference"}
    FAULT_LABELS = {
        "excessive_power_reduction": "BACKHAUL_LOSS / POWER_REDUCTION",
        "interference":              "INTERFERENCE",
    }

    incidents = []
    seen_scenarios = set()
    model.eval()

    # Run on a representative window around each fault scenario
    for scenario, offset in zip(scenarios, offsets):
        if scenario.scenario_id in seen_scenarios:
            continue
        seen_scenarios.add(scenario.scenario_id)
        check_t = offset + scenario.fault_start_s + 35
        if check_t + window >= len(kpi_data):
            continue

        window_raw  = kpi_data[check_t:check_t + window]
        window_norm = (window_raw - mn) / denom
        x = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, learned_adj = model(x, prior)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()

        ts_str = (dataset["base_dt"].strftime("%Y-%m-%dT%H:%M:%SZ"))

        print(f"\n  Scenario: {scenario.name}")
        print(f"  {'Cell':30s} {'Pred':30s} {'Conf':8s} {'True':12s} {'Match'}")
        print(f"  {'─'*30} {'─'*30} {'─'*8} {'─'*12} {'─'*5}")

        for ci in range(N_CELLS):
            cell     = ALL_CELLS[ci]
            pred_cls = int(probs[ci].argmax())
            conf     = float(probs[ci][pred_cls])
            true_cls = int(labels[check_t + window - 1, ci])
            pred_name = FAULT_NAMES[pred_cls]
            true_name = FAULT_NAMES[true_cls]

            if pred_cls > 0 and conf >= 0.50:
                match = "✓" if pred_cls == true_cls else "✗"
                print(f"  {cell.id:30s} {pred_name:30s} "
                      f"{conf:8.3f} {true_name:12s} {match}")

                # Find correlated alarm
                corr_alarm = None
                for alm in normalised_alarms:
                    if (alm["alarmedObject"]["id"] == cell.gnb_id and
                            alm["x_extensions"]["x_scenario_id"] == scenario.scenario_id):
                        corr_alarm = alm
                        break

                incidents.append({
                    "incident_id":       f"SIMBA-{scenario.scenario_id}-C{ci}",
                    "cell_id":           ci,
                    "cell_neo4j_id":     cell.gnb_id,
                    "gnb_id":            cell.gnb_id,
                    "fault_type":        pred_name,
                    "confidence":        round(conf, 4),
                    "severity":          "major" if conf > 0.75 else "minor",
                    "p_normal":          round(float(probs[ci][0]), 4),
                    "p_power":           round(float(probs[ci][1]), 4),
                    "p_interference":    round(float(probs[ci][2]), 4),
                    "timestamp":         ts_str,
                    "window_start":      ts_str,
                    "window_end":        ts_str,
                    "is_root_cause":     ci == scenario.kpi_faults[0][0],
                    "correlated_alarm_id": corr_alarm["id"] if corr_alarm else None,
                    "correlated_alarm_device": corr_alarm["alarmedObject"]["id"] if corr_alarm else None,
                    "scenario_id":       scenario.scenario_id,
                    "kpi_snapshot": {
                        "rsrp_dbm":  round(float(kpi_data[check_t, ci, 0]), 2),
                        "sinr_db":   round(float(kpi_data[check_t, ci, 2]), 2),
                        "dl_tp":     round(float(kpi_data[check_t, ci, 3]), 2),
                        "dl_bler":   round(float(kpi_data[check_t, ci, 5]), 2),
                    },
                    "repair_action": (
                        "Check TX power configuration. Verify RRH and backhaul transport."
                        if pred_cls == 1 else
                        "Activate interference cancellation. Check for rogue transmitters."
                    ),
                })

    print(f"\n  Total SIMBA incidents detected: {len(incidents)}")
    return incidents


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Write to Neo4j
# ─────────────────────────────────────────────────────────────────────────────

def write_to_neo4j(normalised_alarms: list, simba_incidents: list,
                   uri: str, user: str, password: str):
    print_banner("Step 6 — Writing Results to Neo4j")

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("  neo4j not installed. Run: pip install neo4j")
        return

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        print(f"  Connected to Neo4j at {uri}")
    except Exception as e:
        print(f"  Cannot connect to Neo4j: {e}")
        print("  Skipping Neo4j write. Run the demo with Neo4j available.")
        return

    with driver.session() as session:
        # Clear previous incidents and alarms
        session.run("MATCH (i:Incident) DETACH DELETE i")
        session.run("MATCH (a:NormalisedAlarm) DETACH DELETE a")
        session.run("MATCH (a:Alarm) DETACH DELETE a")
        print("  Cleared previous Incident, Alarms and NormalisedAlarm nodes.")

        # Write normalised alarms
        for alm in normalised_alarms:
            session.run("""
                MERGE (a:NormalisedAlarm {id: $id})
                SET a.externalId     = $ext_id,
                    a.alarmType      = $alarm_type,
                    a.severity       = $severity,
                    a.probableCause  = $pc,
                    a.specificProblem= $sp,
                    a.domain         = $domain,
                    a.vendor         = $vendor,
                    a.raisedTime     = $raised,
                    a.isRootCause    = $root,
                    a.source         = 'TMF642-Normalizer'
            """,
                id         = alm["id"],
                ext_id     = alm["externalAlarmId"],
                alarm_type = alm["alarmType"],
                severity   = alm["perceivedSeverity"],
                pc         = alm["probableCause"],
                sp         = alm["specificProblem"],
                domain     = alm["x_extensions"]["x_domain"],
                vendor     = alm["x_extensions"].get("x_vendor",""),
                raised     = alm["alarmRaisedTime"],
                root       = alm["x_extensions"]["x_is_root_cause"],
            )
            # Link to topology node
            device_id = alm["alarmedObject"]["id"]
            session.run("""
                MATCH (a:NormalisedAlarm {id: $alm_id})
                MATCH (n) WHERE n.id = $device_id
                MERGE (a)-[:TRIGGERED_ON]->(n)
            """, alm_id=alm["id"], device_id=device_id)

        print(f"  Written {len(normalised_alarms)} normalised alarms.")

        # Write propagation edges between alarms
        prop_edges = 0
        for alm in normalised_alarms:
            parent_ext = alm["x_extensions"].get("x_propagated_from")
            if parent_ext:
                session.run("""
                    MATCH (parent:NormalisedAlarm {externalId: $parent_ext})
                    MATCH (child:NormalisedAlarm  {id: $child_id})
                    MERGE (parent)-[:PROPAGATED_TO]->(child)
                """, parent_ext=parent_ext, child_id=alm["id"])
                prop_edges += 1
        print(f"  Written {prop_edges} alarm propagation edges.")

        # Write SIMBA incidents
        for inc in simba_incidents:
            session.run("""
                MERGE (i:Incident {id: $id})
                SET i.faultType       = $fault_type,
                    i.confidence      = $confidence,
                    i.severity        = $severity,
                    i.pNormal         = $p_normal,
                    i.pPower          = $p_power,
                    i.pInterference   = $p_int,
                    i.timestamp       = $ts,
                    i.isRootCause     = $root,
                    i.scenarioId      = $scenario_id,
                    i.rsrp            = $rsrp,
                    i.sinr            = $sinr,
                    i.dlThroughput    = $dl_tp,
                    i.dlBler          = $dl_bler,
                    i.repairAction    = $repair,
                    i.source          = 'SIMBA'
            """,
                id          = inc["incident_id"],
                fault_type  = inc["fault_type"],
                confidence  = inc["confidence"],
                severity    = inc["severity"],
                p_normal    = inc["p_normal"],
                p_power     = inc["p_power"],
                p_int       = inc["p_interference"],
                ts          = inc["timestamp"],
                root        = inc["is_root_cause"],
                scenario_id = inc["scenario_id"],
                rsrp        = inc["kpi_snapshot"]["rsrp_dbm"],
                sinr        = inc["kpi_snapshot"]["sinr_db"],
                dl_tp       = inc["kpi_snapshot"]["dl_tp"],
                dl_bler     = inc["kpi_snapshot"]["dl_bler"],
                repair      = inc["repair_action"],
            )
            # Link to RANNode
            session.run("""
                MATCH (i:Incident {id: $inc_id})
                MATCH (n:RANNode {id: $node_id})
                MERGE (i)-[:TRIGGERED_ON]->(n)
            """, inc_id=inc["incident_id"], node_id=inc["cell_neo4j_id"])

            # Cross-link to correlated TMF642 alarm
            if inc.get("correlated_alarm_id"):
                session.run("""
                    MATCH (i:Incident       {id: $inc_id})
                    MATCH (a:NormalisedAlarm {id: $alm_id})
                    MERGE (i)-[:CORRELATED_WITH]->(a)
                """, inc_id=inc["incident_id"],
                     alm_id=inc["correlated_alarm_id"])

        print(f"  Written {len(simba_incidents)} SIMBA incidents.")
        print(f"  Cross-linked SIMBA incidents to TMF642 alarms via CORRELATED_WITH edges.")

    driver.close()
    _print_neo4j_queries()


def _print_neo4j_queries():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  Neo4j Browser queries                                          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Full graph — topology + alarms + SIMBA incidents             ║
║  MATCH (n)-[r]->(m)                                              ║
║  WHERE NOT n:NetworkSlice AND NOT m:NetworkSlice                  ║
║  RETURN n, r, m                                                   ║
║                                                                  ║
║  2. TMF642 alarm propagation chain (fiber cut cascade)           ║
║  MATCH path = (root:NormalisedAlarm {isRootCause:true})           ║
║    -[:PROPAGATED_TO*1..4]->(symptom:NormalisedAlarm)              ║
║  RETURN path                                                      ║
║                                                                  ║
║  3. SIMBA incidents on RAN nodes                                  ║
║  MATCH (i:Incident)-[:TRIGGERED_ON]->(n:RANNode)                  ║
║  RETURN i, n                                                      ║
║                                                                  ║
║  4. Cross-domain correlation — SIMBA + TMF642 on same device     ║
║  MATCH (i:Incident)-[:CORRELATED_WITH]->(a:NormalisedAlarm)       ║
║  MATCH (i)-[:TRIGGERED_ON]->(ran:RANNode)                         ║
║  MATCH (a)-[:TRIGGERED_ON]->(ne)                                  ║
║  RETURN i, a, ran, ne                                             ║
║                                                                  ║
║  5. Root cause to symptom — full cross-domain chain              ║
║  MATCH path =                                                     ║
║    (root:NormalisedAlarm {isRootCause:true})                      ║
║    -[:PROPAGATED_TO*0..4]->(a:NormalisedAlarm)                    ║
║    -[:TRIGGERED_ON]->(ne)                                         ║
║  OPTIONAL MATCH (i:Incident)-[:TRIGGERED_ON]->(ran:RANNode)       ║
║  WHERE ran.id CONTAINS ne.city OR i.scenarioId = a.externalId    ║
║  RETURN path, i, ran                                              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Integrated AIOps Demo")
    parser.add_argument("--quick",      action="store_true",
                        help="Small dataset, fast training for demo")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, use saved model")
    parser.add_argument("--neo4j-uri",  default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="",
                        help="Neo4j password (optional — skip Neo4j if empty)")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    model_path = "models/simba_integrated.pt"
    duration   = 1800 if args.quick else 3600

    # Step 1
    print_topology_summary()

    # Step 2
    dataset = generate_dataset(duration)

    # Step 3 — normalise only unique alarms for display/Neo4j
    seen = set()
    unique_alarm_events = []
    for alarm in dataset["alarm_events"]:
        if alarm.alarm_id not in seen:
            seen.add(alarm.alarm_id)
            unique_alarm_events.append(alarm)
    normalised_alarms = normalise_alarms(unique_alarm_events)

    input("\n  ▶   Press Enter to start training...")
    
    # Step 4
    window = 20 if args.quick else 30
    if not args.skip_train:
        model, norm_params, adj_np = train_simba(dataset, args, model_path)
    else:
        print_banner("Step 4 — Loading saved SIMBA model")
        from simba_pipeline.models.simba import Simba
        adj_np  = build_ran_adjacency()
        ckpt    = torch.load(model_path, map_location="cpu")
        state   = ckpt["model_state"]
        hidden  = state["gcn.input_proj.weight"].shape[0]
        t_layers = sum(1 for k in state
                       if k.startswith("transformer_branch.transformer.layers.")
                       and k.endswith(".norm1.weight"))
        model = Simba(
            n_kpis=len(KPI_NAMES), n_cells=N_CELLS, window_size=window,
            gcn_hidden=hidden, gcn_output=hidden, temporal_dim=hidden,
            n_heads=4, transformer_layers=t_layers,
            ff_dim=hidden*2, fusion_hidden=hidden*2,
        )
        model.load_state_dict(state)
        kpi_flat  = dataset["kpi_data"].reshape(-1, len(KPI_NAMES))
        mn        = kpi_flat.min(axis=0)
        denom_arr = kpi_flat.max(axis=0) - mn
        denom_arr = np.where(denom_arr == 0, 1.0, denom_arr)
        norm_params = (mn, denom_arr)
        print(f"  Loaded model from {model_path}")
    input("\n  ▶   Press Enter to run live inference...")
    # Step 5
    simba_incidents = run_inference_and_correlate(
        model, norm_params, adj_np, dataset, normalised_alarms, window
    )
    input("\n  ▶   Press Enter to write incidents to Neo4j...")
    # Step 6
    if args.neo4j_password:
        write_to_neo4j(
            normalised_alarms, simba_incidents,
            args.neo4j_uri, args.neo4j_user, args.neo4j_password
        )
    else:
        print_banner("Step 6 — Neo4j (skipped — no password provided)")
        print("  To write results to Neo4j, run with:")
        print("  python run_integrated_demo.py --neo4j-password YOUR_PASSWORD")

    # Final summary
    print_banner("Demo Complete — Summary")
    print(f"  Topology        : {len(OPTICAL_NODES)} optical + {len(IP_NODES)} IP"
          f" + {len(GNB_LIST)} gNBs + {N_CELLS} cells")
    print(f"  Fault scenarios : 3 (fiber cut, RRH fault, interference)")
    print(f"  Alarm events    : {len(normalised_alarms)} normalised to TMF642")
    print(f"  SIMBA incidents : {len(simba_incidents)} detected")
    print(f"  Cross-domain    : SIMBA incidents linked to TMF642 alarms in Neo4j")
    print()
    print("  The fiber cut scenario demonstrates the key value proposition:")
    print("  One root cause (optical LOS) drives:")
    print("    → TMF642 alarm normalizer: 7 alarms across 3 vendors")
    print("    → SIMBA: KPI anomaly on 6 affected cells")
    print("    → Neo4j: Full cross-domain causal chain visible in one graph query")


if __name__ == "__main__":
    main()
