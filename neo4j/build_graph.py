"""
Realistic Telecom Topology + Alarm Graph Builder for Neo4j
==========================================================
Builds a full multi-domain telco topology with:
  - Optical layer  : ROADMs, amplifiers, fiber spans
  - IP/MPLS layer  : Core routers, PE routers, agg switches
  - RAN layer      : gNBs, cells (NR + LTE)
  - Compute layer  : Physical hosts, VMs, VNFs (AMF, SMF, UPF)
  - Service layer  : Network slices, services, customers
  - Alarm nodes    : Linked to topology via TRIGGERED_ON edges

Each node carries ONLY its domain-specific label (OpticalNode, IPNode,
RANNode, Host) so Neo4j Browser colour-coding works correctly.

Usage:
    pip install neo4j
    python build_graph.py --uri bolt://YOUR_GCP_IP:7687 --password YOUR_PASSWORD

    # To wipe and rebuild from scratch:
    python build_graph.py --uri bolt://YOUR_GCP_IP:7687 --password YOUR_PASSWORD --reset
"""

import argparse
from datetime import datetime, timedelta
from neo4j import GraphDatabase
import random
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────────────────────────────────────

def get_driver(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print(f"Connected to Neo4j at {uri}")
    return driver


# ─────────────────────────────────────────────────────────────────────────────
# Schema — constraints and indexes
# ─────────────────────────────────────────────────────────────────────────────

CONSTRAINTS = [
    "CREATE CONSTRAINT optical_id   IF NOT EXISTS FOR (n:OpticalNode) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT ip_id        IF NOT EXISTS FOR (n:IPNode)      REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT ran_id       IF NOT EXISTS FOR (n:RANNode)     REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT host_id_c    IF NOT EXISTS FOR (n:Host)        REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT alarm_id     IF NOT EXISTS FOR (a:Alarm)       REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT service_id   IF NOT EXISTS FOR (s:Service)     REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT cell_id      IF NOT EXISTS FOR (c:Cell)        REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT vnf_id       IF NOT EXISTS FOR (v:VNF)         REQUIRE v.id IS UNIQUE",
    "CREATE CONSTRAINT slice_id     IF NOT EXISTS FOR (sl:NetworkSlice) REQUIRE sl.id IS UNIQUE",
    "CREATE INDEX alarm_severity    IF NOT EXISTS FOR (a:Alarm)       ON (a.perceivedSeverity)",
    "CREATE INDEX alarm_domain      IF NOT EXISTS FOR (a:Alarm)       ON (a.domain)",
    "CREATE INDEX alarm_state       IF NOT EXISTS FOR (a:Alarm)       ON (a.state)",
]

def create_schema(session):
    print("Creating schema constraints and indexes...")
    for stmt in CONSTRAINTS:
        try:
            session.run(stmt)
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"  Warning: {e}")
    print("  Schema ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Topology data
# ─────────────────────────────────────────────────────────────────────────────

# ── Optical nodes ─────────────────────────────────────────────────────────
OPTICAL_NODES = [
    {"id": "ROADM-MUM-01", "name": "ROADM Mumbai 01",          "type": "ROADM",
     "vendor": "Nokia",   "model": "1830PSS-32", "site": "Mumbai-POP-1",
     "city": "Mumbai",    "region": "West",      "ip": "10.10.1.1",  "domain": "optical"},
    {"id": "ROADM-MUM-02", "name": "ROADM Mumbai 02",          "type": "ROADM",
     "vendor": "Nokia",   "model": "1830PSS-32", "site": "Mumbai-POP-2",
     "city": "Mumbai",    "region": "West",      "ip": "10.10.1.2",  "domain": "optical"},
    {"id": "ROADM-CHN-01", "name": "ROADM Chennai 01",         "type": "ROADM",
     "vendor": "Nokia",   "model": "1830PSS-32", "site": "Chennai-POP-1",
     "city": "Chennai",   "region": "South",     "ip": "10.10.2.1",  "domain": "optical"},
    {"id": "ROADM-CHN-02", "name": "ROADM Chennai 02",         "type": "ROADM",
     "vendor": "Nokia",   "model": "1830PSS-32", "site": "Chennai-POP-2",
     "city": "Chennai",   "region": "South",     "ip": "10.10.2.2",  "domain": "optical"},
    {"id": "ROADM-BLR-01", "name": "ROADM Bangalore 01",       "type": "ROADM",
     "vendor": "Nokia",   "model": "1830PSS-32", "site": "Bangalore-POP-1",
     "city": "Bangalore", "region": "South",     "ip": "10.10.3.1",  "domain": "optical"},
    {"id": "ROADM-BLR-02", "name": "ROADM Bangalore 02",       "type": "ROADM",
     "vendor": "Nokia",   "model": "1830PSS-32", "site": "Bangalore-POP-2",
     "city": "Bangalore", "region": "South",     "ip": "10.10.3.2",  "domain": "optical"},
    {"id": "ROADM-DEL-01", "name": "ROADM Delhi 01",           "type": "ROADM",
     "vendor": "Ciena",   "model": "6500-T32",   "site": "Delhi-POP-1",
     "city": "Delhi",     "region": "North",     "ip": "10.10.4.1",  "domain": "optical"},
    {"id": "OTN-MUM-01",   "name": "OTN Transponder Mumbai 01","type": "OTN_TRANSPONDER",
     "vendor": "Nokia",   "model": "1830PSS-TXP","site": "Mumbai-POP-1",
     "city": "Mumbai",    "region": "West",      "ip": "10.10.1.11", "domain": "optical"},
    {"id": "OTN-CHN-01",   "name": "OTN Transponder Chennai 01","type": "OTN_TRANSPONDER",
     "vendor": "Nokia",   "model": "1830PSS-TXP","site": "Chennai-POP-1",
     "city": "Chennai",   "region": "South",     "ip": "10.10.2.11", "domain": "optical"},
    {"id": "AMP-MUM-CHN-01","name": "EDFA Amplifier Mum-Chn Span1","type": "AMPLIFIER",
     "vendor": "Nokia",   "model": "1830PSS-AMP","site": "Pune-ILA-1",
     "city": "Pune",      "region": "West",      "ip": "10.10.5.1",  "domain": "optical"},
    {"id": "AMP-MUM-CHN-02","name": "EDFA Amplifier Mum-Chn Span2","type": "AMPLIFIER",
     "vendor": "Nokia",   "model": "1830PSS-AMP","site": "Hyderabad-ILA-1",
     "city": "Hyderabad", "region": "South",     "ip": "10.10.5.2",  "domain": "optical"},
]

# ── IP/MPLS nodes ─────────────────────────────────────────────────────────
IP_NODES = [
    {"id": "RTR-P-CORE-01", "name": "P Router Core 01",       "type": "P_ROUTER",
     "vendor": "Cisco",   "model": "ASR9912",   "site": "Mumbai-DC1",
     "city": "Mumbai",    "region": "West",     "ip": "10.0.0.1",  "asn": 65001, "domain": "ip"},
    {"id": "RTR-P-CORE-02", "name": "P Router Core 02",       "type": "P_ROUTER",
     "vendor": "Cisco",   "model": "ASR9912",   "site": "Chennai-DC1",
     "city": "Chennai",   "region": "South",    "ip": "10.0.0.2",  "asn": 65001, "domain": "ip"},
    {"id": "RTR-P-CORE-03", "name": "P Router Core 03",       "type": "P_ROUTER",
     "vendor": "Huawei",  "model": "NE9000",    "site": "Delhi-DC1",
     "city": "Delhi",     "region": "North",    "ip": "10.0.0.3",  "asn": 65001, "domain": "ip"},
    {"id": "RTR-PE-MUM-01", "name": "PE Router Mumbai 01",    "type": "PE_ROUTER",
     "vendor": "Cisco",   "model": "ASR9006",   "site": "Mumbai-POP-1",
     "city": "Mumbai",    "region": "West",     "ip": "10.1.1.1",  "asn": 65001, "domain": "ip"},
    {"id": "RTR-PE-MUM-02", "name": "PE Router Mumbai 02",    "type": "PE_ROUTER",
     "vendor": "Cisco",   "model": "ASR9006",   "site": "Mumbai-POP-2",
     "city": "Mumbai",    "region": "West",     "ip": "10.1.1.2",  "asn": 65001, "domain": "ip"},
    {"id": "RTR-PE-CHN-01", "name": "PE Router Chennai 01",   "type": "PE_ROUTER",
     "vendor": "Cisco",   "model": "ASR9006",   "site": "Chennai-POP-1",
     "city": "Chennai",   "region": "South",    "ip": "10.1.2.1",  "asn": 65001, "domain": "ip"},
    {"id": "RTR-PE-CHN-02", "name": "PE Router Chennai 02",   "type": "PE_ROUTER",
     "vendor": "Huawei",  "model": "NE40E",     "site": "Chennai-POP-2",
     "city": "Chennai",   "region": "South",    "ip": "10.1.2.2",  "asn": 65001, "domain": "ip"},
    {"id": "RTR-PE-BLR-01", "name": "PE Router Bangalore 01", "type": "PE_ROUTER",
     "vendor": "Cisco",   "model": "ASR9006",   "site": "Bangalore-POP-1",
     "city": "Bangalore", "region": "South",    "ip": "10.1.3.1",  "asn": 65001, "domain": "ip"},
    {"id": "RTR-PE-DEL-01", "name": "PE Router Delhi 01",     "type": "PE_ROUTER",
     "vendor": "Huawei",  "model": "NE40E",     "site": "Delhi-POP-1",
     "city": "Delhi",     "region": "North",    "ip": "10.1.4.1",  "asn": 65001, "domain": "ip"},
    {"id": "SW-AGG-DC1-01", "name": "Agg Switch DC1 01",      "type": "AGG_SWITCH",
     "vendor": "Cisco",   "model": "Nexus9508", "site": "Mumbai-DC1",
     "city": "Mumbai",    "region": "West",     "ip": "10.2.1.1",  "domain": "ip"},
    {"id": "SW-AGG-DC1-02", "name": "Agg Switch DC1 02",      "type": "AGG_SWITCH",
     "vendor": "Cisco",   "model": "Nexus9508", "site": "Mumbai-DC1",
     "city": "Mumbai",    "region": "West",     "ip": "10.2.1.2",  "domain": "ip"},
    {"id": "SW-AGG-DC2-01", "name": "Agg Switch DC2 01",      "type": "AGG_SWITCH",
     "vendor": "Cisco",   "model": "Nexus9508", "site": "Chennai-DC1",
     "city": "Chennai",   "region": "South",    "ip": "10.2.2.1",  "domain": "ip"},
]

# ── RAN nodes ─────────────────────────────────────────────────────────────
RAN_NODES = [
    {"id": "gNB-MUM-SITE-A01", "name": "gNB Mumbai Site Alpha 01",    "type": "gNB",
     "vendor": "Nokia",    "model": "AirScale",  "site": "Mumbai-Alpha-01",
     "city": "Mumbai",     "region": "West",     "ip": "192.168.1.1",
     "cell_count": 3,      "bands": "n78,n41",   "domain": "ran"},
    {"id": "gNB-MUM-SITE-A02", "name": "gNB Mumbai Site Alpha 02",    "type": "gNB",
     "vendor": "Nokia",    "model": "AirScale",  "site": "Mumbai-Alpha-02",
     "city": "Mumbai",     "region": "West",     "ip": "192.168.1.2",
     "cell_count": 3,      "bands": "n78,n41",   "domain": "ran"},
    {"id": "gNB-MUM-SITE-B01", "name": "gNB Mumbai Site Beta 01",     "type": "gNB",
     "vendor": "Ericsson", "model": "AIR6449",   "site": "Mumbai-Beta-01",
     "city": "Mumbai",     "region": "West",     "ip": "192.168.1.3",
     "cell_count": 3,      "bands": "n78",       "domain": "ran"},
    {"id": "gNB-CHN-SITE-A01", "name": "gNB Chennai Site Alpha 01",   "type": "gNB",
     "vendor": "Nokia",    "model": "AirScale",  "site": "Chennai-Alpha-01",
     "city": "Chennai",    "region": "South",    "ip": "192.168.2.1",
     "cell_count": 3,      "bands": "n78,n28",   "domain": "ran"},
    {"id": "gNB-CHN-SITE-A02", "name": "gNB Chennai Site Alpha 02",   "type": "gNB",
     "vendor": "Ericsson", "model": "AIR6449",   "site": "Chennai-Alpha-02",
     "city": "Chennai",    "region": "South",    "ip": "192.168.2.2",
     "cell_count": 3,      "bands": "n78",       "domain": "ran"},
    {"id": "gNB-BLR-SITE-A01", "name": "gNB Bangalore Site Alpha 01", "type": "gNB",
     "vendor": "Nokia",    "model": "AirScale",  "site": "Bangalore-Alpha-01",
     "city": "Bangalore",  "region": "South",    "ip": "192.168.3.1",
     "cell_count": 3,      "bands": "n78,n41",   "domain": "ran"},
    {"id": "gNB-DEL-SITE-A01", "name": "gNB Delhi Site Alpha 01",     "type": "gNB",
     "vendor": "Huawei",   "model": "AAU5613",   "site": "Delhi-Alpha-01",
     "city": "Delhi",      "region": "North",    "ip": "192.168.4.1",
     "cell_count": 3,      "bands": "n78,n41",   "domain": "ran"},
]

# ── Compute / cloud nodes ─────────────────────────────────────────────────
COMPUTE_NODES = [
    {"id": "HOST-DC1-01", "name": "Compute Host DC1 01", "type": "PHYSICAL_HOST",
     "vendor": "Dell",  "model": "PowerEdge R750", "site": "Mumbai-DC1",
     "city": "Mumbai",  "region": "West",  "ip": "10.50.1.1",
     "cpu_cores": 64,   "ram_gb": 512,     "rack": "DC1-ROW-A-R01", "domain": "compute"},
    {"id": "HOST-DC1-02", "name": "Compute Host DC1 02", "type": "PHYSICAL_HOST",
     "vendor": "Dell",  "model": "PowerEdge R750", "site": "Mumbai-DC1",
     "city": "Mumbai",  "region": "West",  "ip": "10.50.1.2",
     "cpu_cores": 64,   "ram_gb": 512,     "rack": "DC1-ROW-A-R02", "domain": "compute"},
    {"id": "HOST-DC1-03", "name": "Compute Host DC1 03", "type": "PHYSICAL_HOST",
     "vendor": "HPE",   "model": "ProLiant DL380", "site": "Mumbai-DC1",
     "city": "Mumbai",  "region": "West",  "ip": "10.50.1.3",
     "cpu_cores": 48,   "ram_gb": 384,     "rack": "DC1-ROW-A-R03", "domain": "compute"},
    {"id": "HOST-DC2-01", "name": "Compute Host DC2 01", "type": "PHYSICAL_HOST",
     "vendor": "Dell",  "model": "PowerEdge R750", "site": "Chennai-DC1",
     "city": "Chennai", "region": "South", "ip": "10.50.2.1",
     "cpu_cores": 64,   "ram_gb": 512,     "rack": "DC2-ROW-A-R01", "domain": "compute"},
    {"id": "HOST-DC2-02", "name": "Compute Host DC2 02", "type": "PHYSICAL_HOST",
     "vendor": "HPE",   "model": "ProLiant DL380", "site": "Chennai-DC1",
     "city": "Chennai", "region": "South", "ip": "10.50.2.2",
     "cpu_cores": 48,   "ram_gb": 384,     "rack": "DC2-ROW-A-R02", "domain": "compute"},
]

# ── VNFs ──────────────────────────────────────────────────────────────────
VNFS = [
    {"id": "VNF-AMF-01",  "name": "AMF Instance 01",  "type": "AMF",
     "vendor": "Nokia",   "version": "22.6",  "host_id": "HOST-DC1-01",
     "ip": "10.60.1.1",   "status": "active", "domain": "compute"},
    {"id": "VNF-SMF-01",  "name": "SMF Instance 01",  "type": "SMF",
     "vendor": "Nokia",   "version": "22.6",  "host_id": "HOST-DC1-01",
     "ip": "10.60.1.2",   "status": "active", "domain": "compute"},
    {"id": "VNF-UPF-01",  "name": "UPF Instance 01",  "type": "UPF",
     "vendor": "Nokia",   "version": "22.6",  "host_id": "HOST-DC1-02",
     "ip": "10.60.1.3",   "status": "active", "domain": "compute"},
    {"id": "VNF-UPF-02",  "name": "UPF Instance 02",  "type": "UPF",
     "vendor": "Nokia",   "version": "22.6",  "host_id": "HOST-DC1-03",
     "ip": "10.60.1.4",   "status": "active", "domain": "compute"},
    {"id": "VNF-PCF-01",  "name": "PCF Instance 01",  "type": "PCF",
     "vendor": "Ericsson","version": "21.Q4", "host_id": "HOST-DC2-01",
     "ip": "10.60.2.1",   "status": "active", "domain": "compute"},
    {"id": "VNF-AUSF-01", "name": "AUSF Instance 01", "type": "AUSF",
     "vendor": "Ericsson","version": "21.Q4", "host_id": "HOST-DC2-01",
     "ip": "10.60.2.2",   "status": "active", "domain": "compute"},
    {"id": "VNF-NRF-01",  "name": "NRF Instance 01",  "type": "NRF",
     "vendor": "Nokia",   "version": "22.6",  "host_id": "HOST-DC2-02",
     "ip": "10.60.2.3",   "status": "active", "domain": "compute"},
]

# ── Cells ─────────────────────────────────────────────────────────────────
def make_cells(gnb_id, city, count=3):
    return [
        {"id": f"CELL-{gnb_id}-{i+1}",
         "name": f"Cell {gnb_id} Sector {i+1}",
         "gnb_id": gnb_id,
         "sector": i + 1,
         "pci": random.randint(1, 503),
         "band": "n78",
         "city": city,
         "domain": "ran"}
        for i in range(count)
    ]

# ── Network slices ────────────────────────────────────────────────────────
SLICES = [
    {"id": "SLICE-EMBB-01",  "name": "eMBB Slice West Region",
     "type": "eMBB",   "region": "West",  "sla_latency_ms": 20,
     "sla_throughput_mbps": 100, "customer": "Consumer-Broadband"},
    {"id": "SLICE-URLLC-01", "name": "uRLLC Slice Industry",
     "type": "uRLLC",  "region": "West",  "sla_latency_ms": 1,
     "sla_throughput_mbps": 10,  "customer": "Industrial-IoT"},
    {"id": "SLICE-EMBB-02",  "name": "eMBB Slice South Region",
     "type": "eMBB",   "region": "South", "sla_latency_ms": 20,
     "sla_throughput_mbps": 100, "customer": "Consumer-Broadband"},
    {"id": "SLICE-MMTC-01",  "name": "mMTC Slice IoT",
     "type": "mMTC",   "region": "West",  "sla_latency_ms": 100,
     "sla_throughput_mbps": 1,   "customer": "Smart-City-IoT"},
]

# ── Services ──────────────────────────────────────────────────────────────
SERVICES = [
    {"id": "SVC-VoNR-WEST",       "name": "VoNR Service West",
     "type": "VoNR",  "slice_id": "SLICE-URLLC-01",
     "customer": "Enterprise-A", "priority": "P1"},
    {"id": "SVC-BROADBAND-WEST",  "name": "Broadband Service West",
     "type": "eMBB",  "slice_id": "SLICE-EMBB-01",
     "customer": "Consumer-B2C", "priority": "P2"},
    {"id": "SVC-IOT-SMART",       "name": "Smart City IoT Service",
     "type": "mMTC",  "slice_id": "SLICE-MMTC-01",
     "customer": "MumbaiCorp",   "priority": "P3"},
    {"id": "SVC-BROADBAND-SOUTH", "name": "Broadband Service South",
     "type": "eMBB",  "slice_id": "SLICE-EMBB-02",
     "customer": "Consumer-B2C", "priority": "P2"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Alarm scenarios
# ─────────────────────────────────────────────────────────────────────────────

def ts(minutes_ago=0):
    return (datetime.utcnow() - timedelta(minutes=minutes_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

ALARMS = [
    # ── Scenario 1: Fiber cut cascade (Optical → IP → RAN) ─────────────────
    {
        "id": "ALM-OPT-001",
        "externalAlarmId": "9876543",
        "alarmType": "communicationsAlarm",
        "perceivedSeverity": "critical",
        "probableCause": "lossOfSignal",
        "specificProblem": "LOS — Optical Channel OCH-1-1-1-TX",
        "alarmDetails": "Loss of Signal on fiber span Mumbai-Chennai. Rx Power: -40.0 dBm",
        "state": "raised",
        "domain": "optical",
        "vendor": "Nokia",
        "sourceSystem": "Nokia-1830PSS",
        "serviceAffecting": True,
        "isRootCause": True,
        "raisedTime": ts(25),
        "triggered_on": "ROADM-MUM-01",
        "affects_service": ["SVC-BROADBAND-WEST", "SVC-VoNR-WEST"],
    },
    {
        "id": "ALM-OPT-002",
        "externalAlarmId": "9876544",
        "alarmType": "equipmentAlarm",
        "perceivedSeverity": "critical",
        "probableCause": "equipmentFailure",
        "specificProblem": "EDFA Amplifier Output Power Below Threshold",
        "alarmDetails": "AMP-MUM-CHN-01 output power: -35 dBm. Expected: +3 dBm.",
        "state": "raised",
        "domain": "optical",
        "vendor": "Nokia",
        "sourceSystem": "Nokia-1830PSS",
        "serviceAffecting": True,
        "isRootCause": False,
        "raisedTime": ts(24),
        "triggered_on": "AMP-MUM-CHN-01",
        "affects_service": [],
    },
    {
        "id": "ALM-IP-001",
        "externalAlarmId": "RTR-PE-MUM-01-LINK-DOWN-20240115",
        "alarmType": "communicationsAlarm",
        "perceivedSeverity": "major",
        "probableCause": "communicationsSubsystemFailure",
        "specificProblem": "LINK-UPDOWN — Interface TenGigE0/0/0/1 down",
        "alarmDetails": "Interface TenGigE0/0/0/1 changed state to down. Peer: RTR-PE-CHN-01.",
        "state": "raised",
        "domain": "ip",
        "vendor": "Cisco",
        "sourceSystem": "Cisco-EPN-Manager",
        "serviceAffecting": True,
        "isRootCause": False,
        "raisedTime": ts(22),
        "triggered_on": "RTR-PE-MUM-01",
        "affects_service": ["SVC-BROADBAND-WEST"],
    },
    {
        "id": "ALM-IP-002",
        "externalAlarmId": "RTR-PE-MUM-01-BGP-DOWN-20240115",
        "alarmType": "communicationsAlarm",
        "perceivedSeverity": "major",
        "probableCause": "softwareProgramAbnormallyTerminated",
        "specificProblem": "BGP-ADJCHG — BGP session to 10.1.2.1 down",
        "alarmDetails": "BGP neighbour 10.1.2.1 (RTR-PE-CHN-01) down. Reason: hold-timer expired.",
        "state": "raised",
        "domain": "ip",
        "vendor": "Cisco",
        "sourceSystem": "Cisco-EPN-Manager",
        "serviceAffecting": True,
        "isRootCause": False,
        "raisedTime": ts(21),
        "triggered_on": "RTR-PE-MUM-01",
        "affects_service": [],
    },
    {
        "id": "ALM-RAN-001",
        "externalAlarmId": "gNB-MUM-SITE-A01-CELL-OOS",
        "alarmType": "communicationsAlarm",
        "perceivedSeverity": "critical",
        "probableCause": "communicationsSubsystemFailure",
        "specificProblem": "NR Cell Out of Service — transport failure",
        "alarmDetails": "5G NR Cells on gNB-MUM-SITE-A01 are out of service. Backhaul N3 path lost.",
        "state": "raised",
        "domain": "ran",
        "vendor": "Nokia",
        "sourceSystem": "Nokia-NetAct",
        "serviceAffecting": True,
        "isRootCause": False,
        "raisedTime": ts(20),
        "triggered_on": "gNB-MUM-SITE-A01",
        "affects_service": ["SVC-BROADBAND-WEST", "SVC-VoNR-WEST"],
    },
    # ── Scenario 2: Compute host failure → VNF down ─────────────────────────
    {
        "id": "ALM-CMP-001",
        "externalAlarmId": "a1b2c3d4e5f6",
        "alarmType": "equipmentAlarm",
        "perceivedSeverity": "critical",
        "probableCause": "equipmentFailure",
        "specificProblem": "HostDown",
        "alarmDetails": "HOST-DC1-02 is unreachable. All VMs on this host are affected.",
        "state": "raised",
        "domain": "compute",
        "vendor": "Prometheus",
        "sourceSystem": "Prometheus-AlertManager",
        "serviceAffecting": True,
        "isRootCause": True,
        "raisedTime": ts(45),
        "triggered_on": "HOST-DC1-02",
        "affects_service": ["SVC-VoNR-WEST"],
    },
    {
        "id": "ALM-CMP-002",
        "externalAlarmId": "upf-pod-0-backoff",
        "alarmType": "processingErrorAlarm",
        "perceivedSeverity": "major",
        "probableCause": "softwareProgramAbnormallyTerminated",
        "specificProblem": "BackOff — UPF pod CrashLoopBackOff",
        "alarmDetails": "UPF-01 pod in CrashLoopBackOff after host failure. Restarted 8 times.",
        "state": "raised",
        "domain": "compute",
        "vendor": "Kubernetes",
        "sourceSystem": "K8s-prod-cluster",
        "serviceAffecting": True,
        "isRootCause": False,
        "raisedTime": ts(44),
        "triggered_on": "HOST-DC1-02",
        "affects_service": ["SVC-VoNR-WEST"],
    },
    # ── Scenario 3: RAN equipment fault ────────────────────────────────────
    {
        "id": "ALM-RAN-002",
        "externalAlarmId": "gNB-CHN-SITE-A01-RRH-FAULT",
        "alarmType": "equipmentAlarm",
        "perceivedSeverity": "major",
        "probableCause": "equipmentFailure",
        "specificProblem": "Radio Unit Hardware Fault",
        "alarmDetails": "RRH unit temperature exceeded 85C on gNB-CHN-SITE-A01 Sector 2.",
        "state": "raised",
        "domain": "ran",
        "vendor": "Nokia",
        "sourceSystem": "Nokia-NetAct",
        "serviceAffecting": True,
        "isRootCause": True,
        "raisedTime": ts(10),
        "triggered_on": "gNB-CHN-SITE-A01",
        "affects_service": ["SVC-BROADBAND-SOUTH"],
    },
    # ── Scenario 4: IP performance degradation ─────────────────────────────
    {
        "id": "ALM-IP-003",
        "externalAlarmId": "HW-2024011500056800",
        "alarmType": "qualityOfServiceAlarm",
        "perceivedSeverity": "major",
        "probableCause": "processorProblem",
        "specificProblem": "CPU Utilization High",
        "alarmDetails": "CPU utilization on RTR-P-CORE-01 exceeded 90% for 5 min. Top process: BGP.",
        "state": "raised",
        "domain": "ip",
        "vendor": "Cisco",
        "sourceSystem": "Cisco-EPN-Manager",
        "serviceAffecting": False,
        "isRootCause": True,
        "raisedTime": ts(5),
        "triggered_on": "RTR-P-CORE-01",
        "affects_service": [],
    },
    # ── Cleared alarm to show lifecycle ────────────────────────────────────
    {
        "id": "ALM-OPT-003",
        "externalAlarmId": "9876100",
        "alarmType": "communicationsAlarm",
        "perceivedSeverity": "cleared",
        "probableCause": "signalQualityEvaluationFailure",
        "specificProblem": "OSNR Degradation — Resolved",
        "alarmDetails": "OSNR on OCH-2-1-1-RX restored to normal. OSNR: 21.4 dB.",
        "state": "cleared",
        "domain": "optical",
        "vendor": "Nokia",
        "sourceSystem": "Nokia-1830PSS",
        "serviceAffecting": False,
        "isRootCause": False,
        "raisedTime": ts(120),
        "cleared_time": ts(90),
        "triggered_on": "ROADM-BLR-01",
        "affects_service": [],
    },
]

# ── Propagation edges (root cause → symptom) ──────────────────────────────
PROPAGATION_EDGES = [
    ("ALM-OPT-001", "ALM-OPT-002"),
    ("ALM-OPT-001", "ALM-IP-001"),
    ("ALM-IP-001",  "ALM-IP-002"),
    ("ALM-IP-001",  "ALM-RAN-001"),
    ("ALM-CMP-001", "ALM-CMP-002"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction functions
# ─────────────────────────────────────────────────────────────────────────────

def create_optical_nodes(session):
    print("Creating optical layer nodes...")
    for n in OPTICAL_NODES:
        session.run("""
            MERGE (node:OpticalNode {id: $id})
            SET node += $props
        """, id=n["id"], props=n)
    print(f"  Created {len(OPTICAL_NODES)} optical nodes.")


def create_ip_nodes(session):
    print("Creating IP/MPLS layer nodes...")
    for n in IP_NODES:
        session.run("""
            MERGE (node:IPNode {id: $id})
            SET node += $props
        """, id=n["id"], props=n)
    print(f"  Created {len(IP_NODES)} IP nodes.")


def create_ran_nodes(session):
    print("Creating RAN layer nodes...")
    for n in RAN_NODES:
        props = {k: v for k, v in n.items()}
        session.run("""
            MERGE (node:RANNode {id: $id})
            SET node += $props
        """, id=n["id"], props=props)

    all_cells = []
    for gnb in RAN_NODES:
        cells = make_cells(gnb["id"], gnb["city"])
        all_cells.extend(cells)
        for cell in cells:
            session.run("""
                MERGE (c:Cell {id: $id})
                SET c += $props
                WITH c
                MATCH (g:RANNode {id: $gnb_id})
                MERGE (c)-[:BELONGS_TO]->(g)
            """, id=cell["id"], props=cell, gnb_id=gnb["id"])

    print(f"  Created {len(RAN_NODES)} gNBs and {len(all_cells)} cells.")


def create_compute_nodes(session):
    print("Creating compute layer nodes...")
    for n in COMPUTE_NODES:
        session.run("""
            MERGE (node:Host {id: $id})
            SET node += $props
        """, id=n["id"], props=n)

    for v in VNFS:
        session.run("""
            MERGE (vnf:VNF {id: $id})
            SET vnf += $props
            WITH vnf
            MATCH (h:Host {id: $host_id})
            MERGE (h)-[:HOSTS]->(vnf)
        """, id=v["id"], props=v, host_id=v["host_id"])

    print(f"  Created {len(COMPUTE_NODES)} hosts and {len(VNFS)} VNFs.")


def create_slices_and_services(session):
    print("Creating slices and services...")
    for sl in SLICES:
        session.run("""
            MERGE (s:NetworkSlice {id: $id})
            SET s += $props
        """, id=sl["id"], props=sl)

    for svc in SERVICES:
        session.run("""
            MERGE (s:Service {id: $id})
            SET s += $props
            WITH s
            MATCH (sl:NetworkSlice {id: $slice_id})
            MERGE (s)-[:BELONGS_TO_SLICE]->(sl)
        """, id=svc["id"], props=svc, slice_id=svc["slice_id"])

    print(f"  Created {len(SLICES)} slices and {len(SERVICES)} services.")


def create_topology_edges(session):
    print("Creating topology edges...")

    # ── Optical fiber spans (ROADM to ROADM) ─────────────────────────────
    fiber_spans = [
        ("ROADM-MUM-01", "ROADM-CHN-01", "MUM-CHN-SPAN-1", "Mumbai-Chennai Fiber Span 1", 1400),
        ("ROADM-MUM-02", "ROADM-CHN-02", "MUM-CHN-SPAN-2", "Mumbai-Chennai Fiber Span 2", 1400),
        ("ROADM-MUM-01", "ROADM-BLR-01", "MUM-BLR-SPAN-1", "Mumbai-Bangalore Fiber Span",  1000),
        ("ROADM-CHN-01", "ROADM-BLR-01", "CHN-BLR-SPAN-1", "Chennai-Bangalore Fiber Span",  350),
        ("ROADM-MUM-01", "ROADM-DEL-01", "MUM-DEL-SPAN-1", "Mumbai-Delhi Fiber Span",       1400),
        ("ROADM-MUM-01", "ROADM-MUM-02", "MUM-INTRA-01",   "Mumbai Intra-city Span",           30),
        ("ROADM-CHN-01", "ROADM-CHN-02", "CHN-INTRA-01",   "Chennai Intra-city Span",           20),
    ]
    for src, dst, span_id, name, km in fiber_spans:
        session.run("""
            MATCH (a:OpticalNode {id: $src})
            MATCH (b:OpticalNode {id: $dst})
            MERGE (a)-[r:FIBER_SPAN {id: $span_id}]->(b)
            SET r.name = $name, r.distance_km = $km, r.type = 'fiber'
            MERGE (b)-[r2:FIBER_SPAN {id: $span_id + '_REV'}]->(a)
            SET r2.name = $name, r2.distance_km = $km, r2.type = 'fiber'
        """, src=src, dst=dst, span_id=span_id, name=name, km=km)

    # ── Amplifiers inline on spans ────────────────────────────────────────
    session.run("""
        MATCH (a:OpticalNode {id: 'ROADM-MUM-01'})
        MATCH (amp:OpticalNode {id: 'AMP-MUM-CHN-01'})
        MATCH (b:OpticalNode {id: 'ROADM-CHN-01'})
        MERGE (a)-[:OPTICAL_LINK {capacity_gbps: 800}]->(amp)
        MERGE (amp)-[:OPTICAL_LINK {capacity_gbps: 800}]->(b)
    """)

    # ── OTN transponders connected to ROADMs ──────────────────────────────
    session.run("""
        MATCH (otn:OpticalNode {id: 'OTN-MUM-01'})
        MATCH (roadm:OpticalNode {id: 'ROADM-MUM-01'})
        MERGE (otn)-[:CONNECTED_TO {type: 'client_port'}]->(roadm)
    """)
    session.run("""
        MATCH (otn:OpticalNode {id: 'OTN-CHN-01'})
        MATCH (roadm:OpticalNode {id: 'ROADM-CHN-01'})
        MERGE (otn)-[:CONNECTED_TO {type: 'client_port'}]->(roadm)
    """)

    # ── Optical circuits — ROADM to IP router (underlay) ─────────────────
    optical_to_ip = [
        ("ROADM-MUM-01", "RTR-PE-MUM-01", "OCH-MUM-RTR1",  100),
        ("ROADM-MUM-02", "RTR-PE-MUM-02", "OCH-MUM-RTR2",  100),
        ("ROADM-CHN-01", "RTR-PE-CHN-01", "OCH-CHN-RTR1",  100),
        ("ROADM-CHN-02", "RTR-PE-CHN-02", "OCH-CHN-RTR2",  100),
        ("ROADM-BLR-01", "RTR-PE-BLR-01", "OCH-BLR-RTR1",  100),
        ("ROADM-DEL-01", "RTR-PE-DEL-01", "OCH-DEL-RTR1",  100),
        ("ROADM-MUM-01", "RTR-P-CORE-01", "OCH-MUM-CORE1", 400),
        ("ROADM-CHN-01", "RTR-P-CORE-02", "OCH-CHN-CORE1", 400),
    ]
    for roadm, rtr, circuit_id, cap in optical_to_ip:
        session.run("""
            MATCH (o:OpticalNode {id: $roadm})
            MATCH (r:IPNode {id: $rtr})
            MERGE (o)-[e:OPTICAL_CIRCUIT {id: $circuit_id}]->(r)
            SET e.capacity_gbps = $cap
        """, roadm=roadm, rtr=rtr, circuit_id=circuit_id, cap=cap)

    # ── IP peering links (router to router) ──────────────────────────────
    ip_links = [
        ("RTR-P-CORE-01", "RTR-P-CORE-02", "IP-CORE-01-02",  400, "MPLS"),
        ("RTR-P-CORE-01", "RTR-P-CORE-03", "IP-CORE-01-03",  400, "MPLS"),
        ("RTR-P-CORE-02", "RTR-P-CORE-03", "IP-CORE-02-03",  400, "MPLS"),
        ("RTR-P-CORE-01", "RTR-PE-MUM-01", "IP-CORE1-PE-MUM1", 100, "MPLS"),
        ("RTR-P-CORE-01", "RTR-PE-MUM-02", "IP-CORE1-PE-MUM2", 100, "MPLS"),
        ("RTR-P-CORE-02", "RTR-PE-CHN-01", "IP-CORE2-PE-CHN1", 100, "MPLS"),
        ("RTR-P-CORE-02", "RTR-PE-CHN-02", "IP-CORE2-PE-CHN2", 100, "MPLS"),
        ("RTR-P-CORE-01", "RTR-PE-BLR-01", "IP-CORE1-PE-BLR1", 100, "MPLS"),
        ("RTR-P-CORE-03", "RTR-PE-DEL-01", "IP-CORE3-PE-DEL1", 100, "MPLS"),
        ("RTR-PE-MUM-01", "RTR-PE-MUM-02", "IP-PE-MUM-PEER",    10, "BGP"),
        ("RTR-PE-CHN-01", "RTR-PE-CHN-02", "IP-PE-CHN-PEER",    10, "BGP"),
    ]
    for src, dst, link_id, cap, protocol in ip_links:
        session.run("""
            MATCH (a:IPNode {id: $src})
            MATCH (b:IPNode {id: $dst})
            MERGE (a)-[r:IP_LINK {id: $link_id}]->(b)
            SET r.capacity_gbps = $cap, r.protocol = $protocol
            MERGE (b)-[r2:IP_LINK {id: $link_id + '_REV'}]->(a)
            SET r2.capacity_gbps = $cap, r2.protocol = $protocol
        """, src=src, dst=dst, link_id=link_id, cap=cap, protocol=protocol)

    # ── IP to compute (aggregation switch to hosts) ───────────────────────
    session.run("""
        MATCH (sw:IPNode {id: 'SW-AGG-DC1-01'})
        MATCH (h:Host) WHERE h.id IN ['HOST-DC1-01','HOST-DC1-02','HOST-DC1-03']
        MERGE (sw)-[:CONNECTED_TO {type: 'server_uplink', speed_gbps: 25}]->(h)
    """)
    session.run("""
        MATCH (sw:IPNode {id: 'SW-AGG-DC2-01'})
        MATCH (h:Host) WHERE h.id IN ['HOST-DC2-01','HOST-DC2-02']
        MERGE (sw)-[:CONNECTED_TO {type: 'server_uplink', speed_gbps: 25}]->(h)
    """)
    session.run("""
        MATCH (rtr:IPNode {id: 'RTR-P-CORE-01'})
        MATCH (sw:IPNode) WHERE sw.id IN ['SW-AGG-DC1-01','SW-AGG-DC1-02']
        MERGE (rtr)-[:CONNECTED_TO {type: 'dc_uplink', speed_gbps: 100}]->(sw)
    """)
    session.run("""
        MATCH (rtr:IPNode {id: 'RTR-P-CORE-02'})
        MATCH (sw:IPNode) WHERE sw.id IN ['SW-AGG-DC2-01']
        MERGE (rtr)-[:CONNECTED_TO {type: 'dc_uplink', speed_gbps: 100}]->(sw)
    """)

    # ── RAN backhaul — gNB to PE router ──────────────────────────────────
    ran_backhaul = [
        ("gNB-MUM-SITE-A01", "RTR-PE-MUM-01"),
        ("gNB-MUM-SITE-A02", "RTR-PE-MUM-01"),
        ("gNB-MUM-SITE-B01", "RTR-PE-MUM-02"),
        ("gNB-CHN-SITE-A01", "RTR-PE-CHN-01"),
        ("gNB-CHN-SITE-A02", "RTR-PE-CHN-02"),
        ("gNB-BLR-SITE-A01", "RTR-PE-BLR-01"),
        ("gNB-DEL-SITE-A01", "RTR-PE-DEL-01"),
    ]
    for gnb, rtr in ran_backhaul:
        session.run("""
            MATCH (g:RANNode {id: $gnb})
            MATCH (r:IPNode {id: $rtr})
            MERGE (g)-[:BACKHAUL_VIA {interfaces: 'N2,N3', speed_gbps: 10}]->(r)
        """, gnb=gnb, rtr=rtr)

    # ── gNB to core VNFs ─────────────────────────────────────────────────
    session.run("""
        MATCH (g:RANNode)
        MATCH (amf:VNF {type: 'AMF'})
        MERGE (g)-[:N2_INTERFACE]->(amf)
    """)
    session.run("""
        MATCH (g:RANNode {region: 'West'})
        MATCH (upf:VNF {id: 'VNF-UPF-01'})
        MERGE (g)-[:N3_INTERFACE]->(upf)
    """)

    # ── Services depend on network nodes ─────────────────────────────────
    service_deps = [
        ("SVC-BROADBAND-WEST",  ["RTR-PE-MUM-01", "RTR-P-CORE-01",
                                  "gNB-MUM-SITE-A01", "gNB-MUM-SITE-A02", "ROADM-MUM-01"]),
        ("SVC-VoNR-WEST",       ["RTR-PE-MUM-01", "RTR-P-CORE-01", "gNB-MUM-SITE-A01"]),
        ("SVC-BROADBAND-SOUTH", ["RTR-PE-CHN-01", "RTR-P-CORE-02",
                                  "gNB-CHN-SITE-A01", "ROADM-CHN-01"]),
        ("SVC-IOT-SMART",       ["RTR-PE-MUM-01", "gNB-MUM-SITE-A01", "gNB-MUM-SITE-B01"]),
    ]
    for svc_id, node_ids in service_deps:
        for node_id in node_ids:
            session.run("""
                MATCH (s:Service {id: $svc_id})
                MATCH (n) WHERE n.id = $node_id
                MERGE (s)-[:DEPENDS_ON]->(n)
            """, svc_id=svc_id, node_id=node_id)

    print("  Topology edges created.")


def create_alarms(session):
    print("Creating alarm nodes...")
    for a in ALARMS:
        props = {k: v for k, v in a.items()
                 if k not in ("triggered_on", "affects_service")}
        session.run("""
            MERGE (alarm:Alarm {id: $id})
            SET alarm += $props
        """, id=a["id"], props=props)

        # TRIGGERED_ON edge — match by id without label since node type varies
        session.run("""
            MATCH (alarm:Alarm {id: $alarm_id})
            MATCH (node) WHERE node.id = $node_id
            MERGE (alarm)-[:TRIGGERED_ON]->(node)
        """, alarm_id=a["id"], node_id=a["triggered_on"])

        # AFFECTS_SERVICE edges
        for svc_id in a.get("affects_service", []):
            session.run("""
                MATCH (alarm:Alarm {id: $alarm_id})
                MATCH (svc:Service {id: $svc_id})
                MERGE (alarm)-[:AFFECTS_SERVICE]->(svc)
            """, alarm_id=a["id"], svc_id=svc_id)

    # PROPAGATED_TO edges between alarms
    for src, dst in PROPAGATION_EDGES:
        session.run("""
            MATCH (a:Alarm {id: $src})
            MATCH (b:Alarm {id: $dst})
            MERGE (a)-[:PROPAGATED_TO]->(b)
        """, src=src, dst=dst)

    print(f"  Created {len(ALARMS)} alarms and {len(PROPAGATION_EDGES)} propagation edges.")


def print_summary(session):
    print("\n" + "=" * 60)
    print("GRAPH SUMMARY")
    print("=" * 60)
    results = session.run("""
        MATCH (n) RETURN labels(n) AS label, count(n) AS count
        ORDER BY count DESC
    """)
    for row in results:
        print(f"  {str(row['label']):35s} {row['count']:>5}")
    print()
    results = session.run("""
        MATCH ()-[r]->()
        RETURN type(r) AS rel, count(r) AS count
        ORDER BY count DESC
    """)
    print("  Relationship types:")
    for row in results:
        print(f"    {row['rel']:35s} {row['count']:>5}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build telecom topology + alarm graph in Neo4j"
    )
    parser.add_argument("--uri",      default="bolt://localhost:7687",
                        help="Neo4j bolt URI (e.g. bolt://34.x.x.x:7687)")
    parser.add_argument("--user",     default="neo4j",
                        help="Neo4j username")
    parser.add_argument("--password", required=True,
                        help="Neo4j password")
    parser.add_argument("--reset",    action="store_true",
                        help="Delete all existing nodes and edges before building")
    args = parser.parse_args()

    driver = get_driver(args.uri, args.user, args.password)

    with driver.session() as session:
        if args.reset:
            print("Resetting graph — deleting all nodes and edges...")
            session.run("MATCH (n) DETACH DELETE n")
            print("  Graph cleared.")

        create_schema(session)
        create_optical_nodes(session)
        create_ip_nodes(session)
        create_ran_nodes(session)
        create_compute_nodes(session)
        create_slices_and_services(session)
        create_topology_edges(session)
        create_alarms(session)
        print_summary(session)

    driver.close()
    print("\nDone. Open Neo4j Browser and run:")
    print("  MATCH (n)-[r]->(m) RETURN n, r, m")
    print("  MATCH p=(a:Alarm {isRootCause:true})-[:PROPAGATED_TO*]->(b:Alarm)")
    print("  RETURN p")


if __name__ == "__main__":
    main()
