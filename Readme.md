# Alarm Normaliser — TMF642 Multi-Vendor AIOps Pipeline

A production-grade alarm normalisation, ML anomaly detection, and knowledge-graph correlation platform for multi-domain telecom networks.

- **Standards**: TM Forum TMF642 v4.0 · ITU-T X.733
- **Vendors**: Cisco · Nokia · Ericsson · Huawei · Prometheus · Kubernetes
- **ML model**: SIMBA (GNN + Transformer, arXiv:2406.15638) for 5G RAN root cause analysis
- **Graph**: Neo4j multi-layer topology spanning Optical → IP → RAN → Compute

---

## Architecture

```
Raw alarms (6 vendor formats)
        │
        ▼
┌─────────────────────────────┐
│   alarm_normalizer/         │  TMF642 normalisation pipeline
│   ├── core/                 │  Canonical model, adapter base, pipeline orchestrator
│   ├── adapters/             │  Cisco · Nokia × 2 · Ericsson · Huawei · SNMP · Prometheus · K8s
│   ├── tests/  (22 tests)    │
│   └── demo/                 │
└──────────────┬──────────────┘
               │  CanonicalAlarm stream
               ▼
┌─────────────────────────────┐
│   simba_pipeline/           │  5G RAN anomaly detection
│   ├── models/simba.py       │  GraphStructureLearning + GCN + Transformer
│   ├── data/                 │  Synthetic KPI generator (21-cell hexagonal topology)
│   ├── training/             │  Focal loss · early stopping · LR scheduling
│   ├── inference/            │  Sliding-window real-time engine
│   └── tests/  (59 tests)    │
└──────────────┬──────────────┘
               │  Fault detections + learned adjacency
               ▼
┌─────────────────────────────┐
│   integrated_aiops/         │  End-to-end orchestration
│   ├── topology/             │  Unified topology: 11 optical + 12 IP + 21 RAN + 5 compute nodes
│   ├── scenarios/            │  Cross-domain fault simulation (fiber cut, RRH fault, CPU high)
│   └── run_integrated_demo.py│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   neo4j/                    │  Knowledge graph
│   ├── build_graph.py        │  Full topology + alarm nodes + service mappings
│   └── create_fiber_cut_alarms.py
└─────────────────────────────┘
```

---

## Quick Start

### Alarm normalisation (no ML dependencies)

```bash
git clone https://github.com/aryanshekhar/alarm-normalizer.git
cd alarm-normalizer/alarm_normalizer

# Run test suite (22 tests, pure Python)
python tests/test_pipeline.py

# Interactive demo — 8 vendor scenarios
python demo/run_demo.py

# Single scenario (fiber-cut cascade)
python demo/run_demo.py --scenario fiber

# Show full TMF642 JSON output
python demo/run_demo.py --scenario cisco --json
```

### SIMBA ML pipeline

```bash
pip install torch>=2.0.0 numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0

cd simba_pipeline

# Run test suite (59 tests)
python tests/test_simba.py

# End-to-end demo: generate data → train → infer
python run_demo.py

# Quick run (small dataset, ~30 seconds)
python run_demo.py --quick

# Skip training, use pre-trained model
python run_demo.py --skip-train
```

### Integrated AIOps demo

```bash
pip install -r requirements.txt   # includes neo4j driver

# Full pipeline: topology → faults → alarms → KPIs → SIMBA → Neo4j
# (requires a running Neo4j instance — see Neo4j section below)
cd integrated_aiops
python run_integrated_demo.py
```

---

## Alarm Normalisation Pipeline

### Supported vendors and formats

| Vendor      | Domain   | Format           | Adapter class          |
|-------------|----------|------------------|------------------------|
| Cisco       | IP       | Syslog           | `CiscoSyslogAdapter`   |
| Nokia       | RAN      | NetAct JSON      | `NokiaNetActAdapter`   |
| Nokia       | Optical  | 1830 PSS JSON    | `Nokia1830PSSAdapter`  |
| Ericsson    | RAN      | ENM REST JSON    | `EricssonENMAdapter`   |
| Huawei      | IP / RAN | iManager JSON    | `HuaweiIManagerAdapter`|
| Generic     | IP       | SNMP trap        | `SNMPTrapAdapter`      |
| Prometheus  | Compute  | AlertManager     | `PrometheusAlertAdapter`|
| Kubernetes  | Compute  | K8s Event        | `KubernetesEventAdapter`|

### Using the pipeline

```python
from core.pipeline import NormalisationPipeline

pipeline = NormalisationPipeline()

alarm = pipeline.process(
    raw_payload={"probableCause": "CELL_DISABLED", "perceivedSeverity": "A1", ...},
    vendor="ericsson",
    domain="ran",
    format="json_rest",
)
print(alarm.to_dict())        # TMF642-compliant JSON
print(pipeline.stats)         # processed / valid / deduplicated counts
```

**Injecting a real CMDB** (for site/region enrichment in production):

```python
def my_cmdb(ne_id: str) -> dict:
    return cmdb_api.get(f"/inventory/{ne_id}") or {}

pipeline = NormalisationPipeline(cmdb_lookup=my_cmdb)
```

### Output format

Every alarm is a `CanonicalAlarm` dataclass serialisable to TMF642 JSON:

```json
{
  "id": "3f2a...",
  "alarmRaisedTime": "2024-01-15T14:23:01Z",
  "alarmType": "communicationsAlarm",
  "perceivedSeverity": "critical",
  "state": "raised",
  "alarmedObject": {"id": "RBS-SITE-001", "name": "RBS-SITE-001", "referredType": "EUtranCellFDD"},
  "probableCause": "loss-of-signal",
  "specificProblem": "Cell Disabled",
  "vendor": "Ericsson",
  "networkDomain": "ran",
  "@type": "Alarm",
  "x_extensions": {"rawFormat": "json_rest", "sourceSystemId": "Ericsson-ENM"}
}
```

---

## SIMBA ML Model

SIMBA detects 5G RAN anomalies from KPI time series across 21 cells (7 gNBs × 3 sectors).

**Output classes per cell:**
- `0` — normal
- `1` — excessive power reduction (TX power dropped)
- `2` — interference (external RF interference)

**Architecture:**
```
KPI window (T, N_cells, 9 KPIs)
    ├── Graph Structure Learning  →  learned adjacency (N, N)
    ├── GCN (2 layers)            →  spatial embedding  (B, N, 64)
    └── Transformer (2 layers)    →  temporal embedding (B, N, 64)
                    └── Fusion head → logits (B, N, 3)
```

### Real-time inference

```python
from inference.inference_engine import SimbaInferenceEngine
import numpy as np

engine = SimbaInferenceEngine(
    model_path="models/simba_best.pt",
    normalizer_path="data/kpi_normalizer.npz",
    adjacency=adj_matrix,   # (n_cells, n_cells) physical topology
)

# Feed one timestep per call (n_cells, n_kpis)
result = engine.ingest(kpi_snapshot)
if result:
    for d in result.anomalous_cells:
        print(d.cell_id, d.fault_type, d.confidence, d.repair_action)
```

---

## Neo4j Knowledge Graph

Requires Neo4j 5.x. Start a local instance:

```bash
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.18.0
```

Build the topology graph:

```bash
# Set connection details
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=password

python neo4j/build_graph.py
```

Load the fiber-cut alarm scenario for RCA visualisation:

```bash
python neo4j/create_fiber_cut_alarms.py
```

Open the Neo4j Browser at `http://localhost:7474` and run:

```cypher
MATCH path = (a:Alarm)-[:TRIGGERED_ON]->(n)-[:CONNECTS_TO*1..4]-(m)
RETURN path LIMIT 50
```

---

## Project Structure

```
alarm-normalizer/
├── alarm_normalizer/          # Normalisation pipeline (pure Python, no ML deps)
│   ├── core/                  # model.py · base_adapter.py · pipeline.py
│   ├── adapters/              # One file per vendor family
│   ├── data/test_samples.py   # Real-world alarm samples for all vendors
│   ├── demo/run_demo.py
│   └── tests/test_pipeline.py
├── simba_pipeline/            # ML anomaly detection
│   ├── models/simba.py        # GNN + Transformer architecture
│   ├── data/dataset_generator.py
│   ├── training/train.py
│   ├── inference/inference_engine.py
│   └── tests/test_simba.py
├── integrated_aiops/          # End-to-end orchestration
│   ├── topology/unified_topology.py
│   ├── scenarios/fault_propagation.py
│   └── run_integrated_demo.py
├── neo4j/                     # Knowledge graph builders
├── docs/                      # LLD design documents
└── requirements.txt
```

---

## Requirements

| Component            | Python | Dependencies                                    |
|----------------------|--------|-------------------------------------------------|
| alarm_normalizer     | ≥ 3.8  | None (stdlib only)                              |
| simba_pipeline       | ≥ 3.9  | torch ≥ 2.0, numpy, pandas, scikit-learn        |
| integrated_aiops     | ≥ 3.9  | All above + neo4j == 5.18.0                     |
| Kafka integration    | —      | kafka-python ≥ 2.0.2 (optional)                |

```bash
# Alarm normaliser only
pip install  # nothing required

# Full stack
pip install -r requirements.txt
```

---

## CI

GitHub Actions runs on every push and pull request:
- `alarm_normalizer/tests/test_pipeline.py` — 22 unit tests
- `alarm_normalizer/demo/run_demo.py --scenario fiber` — fiber-cut cascade smoke test
- `simba_pipeline/tests/test_simba.py` — 59 unit tests (model, data, inference)
