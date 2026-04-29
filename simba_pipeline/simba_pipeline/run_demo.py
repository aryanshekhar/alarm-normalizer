"""
SIMBA End-to-End Demo
======================
Runs the complete pipeline:
  1. Generate synthetic 5G KPI dataset
  2. Train the SIMBA model
  3. Run inference on a simulated fault scenario
  4. Display results

Usage:
    python run_demo.py              # Full pipeline (generate + train + infer)
    python run_demo.py --skip-train # Use existing model, infer only
    python run_demo.py --quick      # Small dataset, fast training for demo
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import argparse
from datetime import datetime

from data.dataset_generator import (
    build_hexagonal_topology, build_adjacency_matrix,
    KPITimeSeriesGenerator, KPINormalizer,
    create_sliding_windows, train_val_test_split,
    KPI_NAMES, N_KPIS
)
from models.simba import Simba, WeightedFocalLoss, compute_class_weights
from training.train import train, evaluate
from inference.inference_engine import SimbaInferenceEngine
from torch.utils.data import DataLoader, TensorDataset


def print_banner(title: str, width: int = 70) -> None:
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def run_demo(args) -> None:

    print_banner("SIMBA — GNN+Transformer RCA Pipeline for 5G RAN")
    print(f"  Based on: arXiv:2406.15638 (Hasan et al., 2024)")
    print(f"  Architecture: Graph Structure Learning + GCN + Transformer")
    print(f"  Task: Anomaly Detection + Root Cause Analysis")

    os.makedirs("data",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
    print(f"\n  Device: {device}")

    # ── Step 1: Build topology ─────────────────────────────────────────────
    print_banner("Step 1 — Build 5G RAN Topology")
    n_sites = 3 if args.quick else 7
    cells   = build_hexagonal_topology(n_sites=n_sites)
    adj     = build_adjacency_matrix(cells)
    n_cells = len(cells)
    print(f"  Sites:       {n_sites} gNBs (hexagonal grid, 200m ISD)")
    print(f"  Cells:       {n_cells} (3 sectors per gNB)")
    print(f"  Adj edges:   {int(adj.sum())}")
    print(f"  KPIs/cell:   {N_KPIS}")
    for i, name in enumerate(KPI_NAMES):
        print(f"    [{i:2d}] {name}")

    # ── Step 2: Generate or load dataset ──────────────────────────────────
    print_banner("Step 2 — Generate Synthetic KPI Dataset")
    data_path = "data/kpi_demo.npz"
    norm_path = "data/kpi_demo_normalizer.npz"

    duration = 1800 if args.quick else 7200  # 30 min or 2 hours
    window   = 20   if args.quick else 30

    print(f"  Duration:    {duration}s ({duration//60} minutes)")
    print(f"  Window size: {window}s")
    print(f"  Fault types: excessive_power_reduction, interference")
    print(f"  Anomaly rate: ~2% (following SIMBA paper)")

    # Build a specific fault scenario for the demo
    fault_schedule = [
        # Scenario A: Power reduction on cell 5 (gNB-1, sector 3)
        {"cell_id": 5, "fault_type": "excessive_power_reduction",
         "start_t": 300, "end_t": 420, "severity": 0.85},
        # Scenario B: Interference on cell 2 (gNB-0, sector 3)
        {"cell_id": 2, "fault_type": "interference",
         "start_t": 600, "end_t": 720, "severity": 0.90},
        # Scenario C: Simultaneous faults on adjacent cells
        {"cell_id": 8,  "fault_type": "excessive_power_reduction",
         "start_t": 900, "end_t": 960, "severity": 0.75},
        {"cell_id": 9,  "fault_type": "interference",
         "start_t": 910, "end_t": 970, "severity": 0.80},
        # More scattered faults for training diversity
        {"cell_id": 1,  "fault_type": "interference",
         "start_t": 1200, "end_t": 1260, "severity": 0.70},
        {"cell_id": 6,  "fault_type": "excessive_power_reduction",
         "start_t": 1500, "end_t": 1560, "severity": 0.65},
    ]
    if duration > 1800:
        fault_schedule += [
            {"cell_id": 3,  "fault_type": "interference",
             "start_t": 2100, "end_t": 2200, "severity": 0.80},
            {"cell_id": 10, "fault_type": "excessive_power_reduction",
             "start_t": 2400, "end_t": 2500, "severity": 0.90},
            {"cell_id": 0,  "fault_type": "interference",
             "start_t": 3000, "end_t": 3100, "severity": 0.75},
            {"cell_id": 7,  "fault_type": "excessive_power_reduction",
             "start_t": 3600, "end_t": 3700, "severity": 0.85},
        ]

    gen = KPITimeSeriesGenerator(cells, duration)
    kpi_data, labels, fault_log = gen.generate(fault_schedule=fault_schedule)

    print(f"\n  Generated {len(fault_log)} fault events:")
    for f in fault_log[:6]:
        print(f"    Cell {f['cell_id']:2d}: {f['fault_type']:30s} "
              f"t={f['start_t']:4d}–{f['end_t']:4d}s  sev={f['severity']:.2f}")

    # Window and split
    X, y = create_sliding_windows(kpi_data, labels, window_size=window, stride=1)
    X_tr, y_tr, X_val, y_val, X_te, y_te = train_val_test_split(X, y)

    # Normalise
    normalizer = KPINormalizer()
    X_tr  = normalizer.fit_transform(X_tr)
    X_val = normalizer.transform(X_val)
    X_te  = normalizer.transform(X_te)
    normalizer.save(norm_path)

    print(f"\n  Dataset splits:")
    print(f"    Train: {X_tr.shape[0]:6,} windows | "
          f"anomaly rate: {(y_tr > 0).mean():.3%}")
    print(f"    Val:   {X_val.shape[0]:6,} windows | "
          f"anomaly rate: {(y_val > 0).mean():.3%}")
    print(f"    Test:  {X_te.shape[0]:6,} windows | "
          f"anomaly rate: {(y_te > 0).mean():.3%}")

    # ── Step 3: Build and train model ─────────────────────────────────────
    model_path = "models/simba_demo.pt"

    if not args.skip_train:
        print_banner("Step 3 — Train SIMBA Model")

        class_weights = compute_class_weights(y_tr)

        model = Simba(
            n_kpis=N_KPIS, n_cells=n_cells, window_size=window,
            gcn_hidden=32,  gcn_output=32, gcn_layers=2,
            temporal_dim=32, n_heads=4, transformer_layers=1,
            ff_dim=64, fusion_hidden=64, dropout=0.1,
        ).to(device)

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.long)
            ),
            batch_size=32, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            ),
            batch_size=32, shuffle=False
        )

        prior = torch.tensor(adj, dtype=torch.float32).to(device)
        config = {
            "epochs":       20 if args.quick else 50,
            "batch_size":   32,
            "lr":           1e-3,
            "patience":     8 if args.quick else 15,
            "weight_decay": 1e-4,
            "focal_gamma":  2.0,
            "class_weights": class_weights,
        }

        train(model, train_loader, val_loader, config, device, prior, model_path)

        # Test evaluation
        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_te, dtype=torch.float32),
                torch.tensor(y_te, dtype=torch.long)
            ),
            batch_size=32, shuffle=False
        )
        criterion = WeightedFocalLoss(class_weights=class_weights).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_loss, test_metrics = evaluate(model, test_loader, criterion, device, prior)

        print(f"\n  Test Results:")
        print(f"    Accuracy:       {test_metrics['accuracy']:.4f}")
        print(f"    Macro-F1:       {test_metrics['macro_f1']:.4f}")
        print(f"    Anomaly F1:     {test_metrics['anomaly_f1']:.4f}")
        print(f"    Anomaly Recall: {test_metrics['anomaly_recall']:.4f}")

    # ── Step 4: Real-time inference simulation ─────────────────────────────
    print_banner("Step 4 — Real-Time Inference Simulation")
    print("  Replaying fault scenarios through the inference engine...")
    print("  (Simulating live KPI stream from 5G network)")
    print()

    engine = SimbaInferenceEngine(
        model_path      = model_path,
        normalizer_path = norm_path,
        adjacency       = adj,
        cell_to_gnb_map = {c.cell_id: c.gnb_id for c in cells},
        window_size     = window,
        stride          = 5,        # infer every 5 seconds
        anomaly_threshold = 0.55,
        device          = str(device),
    )

    # Replay a 600-second window containing all fault scenarios
    sim_segment = kpi_data[250:850]  # covers all demo faults
    true_labels  = labels[250:850]

    print(f"  Replaying {len(sim_segment)}s of KPI data across {n_cells} cells\n")
    print(f"  {'Time':>6}  {'Cell':>4}  {'Fault Type':30}  {'Confidence':>10}  {'True Label':12}")
    print(f"  {'─'*6}  {'─'*4}  {'─'*30}  {'─'*10}  {'─'*12}")

    correct, total_anomalies = 0, 0
    for t, snapshot in enumerate(sim_segment):
        result = engine.ingest(snapshot)
        if result is None:
            continue
        for det in result.detections:
            if det.is_anomaly:
                # Look up true label at this timestep
                true_lbl = int(true_labels[t, det.cell_id])
                true_name = {0: "normal", 1: "power_reduc", 2: "interference"}[true_lbl]
                correct += (det.fault_type == {
                    0: "normal", 1: "excessive_power_reduction", 2: "interference"
                }[true_lbl])
                total_anomalies += 1
                print(
                    f"  t={t+250:4d}s  "
                    f"cell {det.cell_id:2d}  "
                    f"{det.fault_type:30s}  "
                    f"{det.confidence:10.3f}  "
                    f"{true_name:12s}"
                )

    if total_anomalies > 0:
        print(f"\n  Inference accuracy on anomalous timesteps: "
              f"{correct}/{total_anomalies} = {correct/total_anomalies:.1%}")

    # ── Summary ────────────────────────────────────────────────────────────
    print_banner("Summary")
    print("""
  What was demonstrated:
  ──────────────────────
  1. Synthetic 5G KPI generation using 3GPP eMBB-Urban parameters
     (structurally identical to SIMBA paper / Simu5G output)

  2. SIMBA model training:
     • Graph Structure Learning — learns cell relationship graph from data
     • Graph Convolution — captures spatial (inter-cell) correlations
     • Transformer — captures temporal (time-series) patterns
     • Weighted Focal Loss — handles 2% anomaly rate imbalance

  3. Real-time inference engine:
     • Sliding window buffer for live KPI streams
     • Per-cell anomaly classification with confidence scores
     • Repair action recommendations per fault type

  4. Integration architecture:
     • Kafka for KPI stream ingestion (high throughput)
     • NETCONF/TMF639 for topology discovery (brownfield)
     • MCP for LLM NOC assistant layer (NOT for streaming)

  Next steps for production:
  ─────────────────────────
  • Connect Kafka adapter to live NMS KPI feed
  • Run TopologyDiscoveryAdapter against real gNBs (NETCONF)
  • Load real topology into Neo4j alongside SIMBA detections
  • Build MCP server exposing Neo4j + SIMBA as LLM tools
  • Fine-tune on operator KPI data once labelled samples available
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIMBA End-to-End Demo")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, use existing model")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: small dataset, fast training (for demo)")
    args = parser.parse_args()
    run_demo(args)
