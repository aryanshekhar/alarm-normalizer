"""
Unified Telecom Topology
========================
Single source of truth for the network topology used by ALL components:
  - Alarm normalisation pipeline (TMF642)
  - SIMBA KPI anomaly detection
  - Neo4j knowledge graph
  - Gephi visualisation

Topology: 7-site Indian telco (Mumbai, Chennai, Bangalore, Delhi)
  Optical  : 11 nodes (ROADMs, amplifiers, OTN transponders)
  IP/MPLS  : 12 nodes (P-routers, PE-routers, agg switches)
  RAN      : 7 gNBs × 3 sectors = 21 cells
  Compute  : 5 hosts, 7 VNFs (5GC: AMF, SMF, UPF×2, PCF, AUSF, NRF)

Cell-to-gNB mapping is explicitly maintained so SIMBA's cell indices
map directly to the same gNB objects in Neo4j and the alarm normalizer.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Node definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpticalNode:
    id: str; name: str; type: str; vendor: str
    model: str; site: str; city: str; region: str; ip: str
    domain: str = "optical"

@dataclass
class IPNode:
    id: str; name: str; type: str; vendor: str
    model: str; site: str; city: str; region: str; ip: str
    domain: str = "ip"

@dataclass
class RANCell:
    """
    A single sector/cell — the atomic unit for SIMBA and alarm normalizer.
    cell_index is the 0-based index used by SIMBA (0..20 for 21 cells).
    """
    cell_index: int        # SIMBA index 0..20
    id: str                # Neo4j / alarm normalizer ID e.g. "gNB-MUM-SITE-A01-S1"
    gnb_id: str            # Parent gNB e.g. "gNB-MUM-SITE-A01"
    sector: int            # 1, 2, or 3
    vendor: str
    city: str
    region: str
    band: str = "n78"
    domain: str = "ran"

@dataclass
class gNB:
    id: str; name: str; vendor: str; model: str
    site: str; city: str; region: str; ip: str
    cells: List[RANCell] = field(default_factory=list)
    backhaul_pe: str = ""  # ID of PE router this gNB connects to
    domain: str = "ran"

@dataclass
class ComputeHost:
    id: str; name: str; vendor: str; model: str
    site: str; city: str; region: str; ip: str
    cpu_cores: int; ram_gb: int; rack: str
    domain: str = "compute"

@dataclass
class VNF:
    id: str; name: str; type: str; vendor: str
    version: str; host_id: str; ip: str; status: str
    domain: str = "compute"


# ─────────────────────────────────────────────────────────────────────────────
# Optical layer
# ─────────────────────────────────────────────────────────────────────────────

OPTICAL_NODES = [
    OpticalNode("ROADM-MUM-01","ROADM Mumbai 01","ROADM","Nokia","1830PSS-32","Mumbai-POP-1","Mumbai","West","10.10.1.1"),
    OpticalNode("ROADM-MUM-02","ROADM Mumbai 02","ROADM","Nokia","1830PSS-32","Mumbai-POP-2","Mumbai","West","10.10.1.2"),
    OpticalNode("ROADM-CHN-01","ROADM Chennai 01","ROADM","Nokia","1830PSS-32","Chennai-POP-1","Chennai","South","10.10.2.1"),
    OpticalNode("ROADM-CHN-02","ROADM Chennai 02","ROADM","Nokia","1830PSS-32","Chennai-POP-2","Chennai","South","10.10.2.2"),
    OpticalNode("ROADM-BLR-01","ROADM Bangalore 01","ROADM","Nokia","1830PSS-32","Bangalore-POP-1","Bangalore","South","10.10.3.1"),
    OpticalNode("ROADM-BLR-02","ROADM Bangalore 02","ROADM","Nokia","1830PSS-32","Bangalore-POP-2","Bangalore","South","10.10.3.2"),
    OpticalNode("ROADM-DEL-01","ROADM Delhi 01","ROADM","Ciena","6500-T32","Delhi-POP-1","Delhi","North","10.10.4.1"),
    OpticalNode("OTN-MUM-01","OTN Transponder Mumbai 01","OTN_TRANSPONDER","Nokia","1830PSS-TXP","Mumbai-POP-1","Mumbai","West","10.10.1.11"),
    OpticalNode("OTN-CHN-01","OTN Transponder Chennai 01","OTN_TRANSPONDER","Nokia","1830PSS-TXP","Chennai-POP-1","Chennai","South","10.10.2.11"),
    OpticalNode("AMP-MUM-CHN-01","EDFA Amplifier Mum-Chn Span1","AMPLIFIER","Nokia","1830PSS-AMP","Pune-ILA-1","Pune","West","10.10.5.1"),
    OpticalNode("AMP-MUM-CHN-02","EDFA Amplifier Mum-Chn Span2","AMPLIFIER","Nokia","1830PSS-AMP","Hyderabad-ILA-1","Hyderabad","South","10.10.5.2"),
]

# ─────────────────────────────────────────────────────────────────────────────
# IP/MPLS layer
# ─────────────────────────────────────────────────────────────────────────────

IP_NODES = [
    IPNode("RTR-P-CORE-01","P Router Core 01","P_ROUTER","Cisco","ASR9912","Mumbai-DC1","Mumbai","West","10.0.0.1"),
    IPNode("RTR-P-CORE-02","P Router Core 02","P_ROUTER","Cisco","ASR9912","Chennai-DC1","Chennai","South","10.0.0.2"),
    IPNode("RTR-P-CORE-03","P Router Core 03","P_ROUTER","Huawei","NE9000","Delhi-DC1","Delhi","North","10.0.0.3"),
    IPNode("RTR-PE-MUM-01","PE Router Mumbai 01","PE_ROUTER","Cisco","ASR9006","Mumbai-POP-1","Mumbai","West","10.1.1.1"),
    IPNode("RTR-PE-MUM-02","PE Router Mumbai 02","PE_ROUTER","Cisco","ASR9006","Mumbai-POP-2","Mumbai","West","10.1.1.2"),
    IPNode("RTR-PE-CHN-01","PE Router Chennai 01","PE_ROUTER","Cisco","ASR9006","Chennai-POP-1","Chennai","South","10.1.2.1"),
    IPNode("RTR-PE-CHN-02","PE Router Chennai 02","PE_ROUTER","Huawei","NE40E","Chennai-POP-2","Chennai","South","10.1.2.2"),
    IPNode("RTR-PE-BLR-01","PE Router Bangalore 01","PE_ROUTER","Cisco","ASR9006","Bangalore-POP-1","Bangalore","South","10.1.3.1"),
    IPNode("RTR-PE-DEL-01","PE Router Delhi 01","PE_ROUTER","Huawei","NE40E","Delhi-POP-1","Delhi","North","10.1.4.1"),
    IPNode("SW-AGG-DC1-01","Agg Switch DC1 01","AGG_SWITCH","Cisco","Nexus9508","Mumbai-DC1","Mumbai","West","10.2.1.1"),
    IPNode("SW-AGG-DC1-02","Agg Switch DC1 02","AGG_SWITCH","Cisco","Nexus9508","Mumbai-DC1","Mumbai","West","10.2.1.2"),
    IPNode("SW-AGG-DC2-01","Agg Switch DC2 01","AGG_SWITCH","Cisco","Nexus9508","Chennai-DC1","Chennai","South","10.2.2.1"),
]

# ─────────────────────────────────────────────────────────────────────────────
# RAN layer — 7 gNBs, 3 sectors each = 21 cells, cell_index 0..20
# ─────────────────────────────────────────────────────────────────────────────

def _make_gnb(gnb_id, name, vendor, model, site, city, region, ip,
              backhaul_pe, cell_index_start):
    """Create a gNB with 3 sector cells, cell indices starting at cell_index_start."""
    cells = [
        RANCell(
            cell_index = cell_index_start + s,
            id         = f"{gnb_id}-S{s+1}",
            gnb_id     = gnb_id,
            sector     = s + 1,
            vendor     = vendor,
            city       = city,
            region     = region,
        )
        for s in range(3)
    ]
    return gNB(
        id=gnb_id, name=name, vendor=vendor, model=model,
        site=site, city=city, region=region, ip=ip,
        cells=cells, backhaul_pe=backhaul_pe
    )

GNB_LIST = [
    _make_gnb("gNB-MUM-SITE-A01","gNB Mumbai Alpha 01","Nokia","AirScale",
              "Mumbai-Alpha-01","Mumbai","West","192.168.1.1","RTR-PE-MUM-01",0),
    _make_gnb("gNB-MUM-SITE-A02","gNB Mumbai Alpha 02","Nokia","AirScale",
              "Mumbai-Alpha-02","Mumbai","West","192.168.1.2","RTR-PE-MUM-01",3),
    _make_gnb("gNB-MUM-SITE-B01","gNB Mumbai Beta 01","Ericsson","AIR6449",
              "Mumbai-Beta-01","Mumbai","West","192.168.1.3","RTR-PE-MUM-02",6),
    _make_gnb("gNB-CHN-SITE-A01","gNB Chennai Alpha 01","Nokia","AirScale",
              "Chennai-Alpha-01","Chennai","South","192.168.2.1","RTR-PE-CHN-01",9),
    _make_gnb("gNB-CHN-SITE-A02","gNB Chennai Alpha 02","Ericsson","AIR6449",
              "Chennai-Alpha-02","Chennai","South","192.168.2.2","RTR-PE-CHN-02",12),
    _make_gnb("gNB-BLR-SITE-A01","gNB Bangalore Alpha 01","Nokia","AirScale",
              "Bangalore-Alpha-01","Bangalore","South","192.168.3.1","RTR-PE-BLR-01",15),
    _make_gnb("gNB-DEL-SITE-A01","gNB Delhi Alpha 01","Huawei","AAU5613",
              "Delhi-Alpha-01","Delhi","North","192.168.4.1","RTR-PE-DEL-01",18),
]

# Flat cell list — cell_index is the canonical reference
ALL_CELLS: List[RANCell] = [cell for gnb in GNB_LIST for cell in gnb.cells]
N_CELLS = len(ALL_CELLS)  # 21

# Convenience lookup maps
CELL_BY_INDEX: Dict[int, RANCell]  = {c.cell_index: c for c in ALL_CELLS}
CELL_BY_ID:    Dict[str, RANCell]  = {c.id: c for c in ALL_CELLS}
GNB_BY_ID:     Dict[str, gNB]     = {g.id: g for g in GNB_LIST}

# ─────────────────────────────────────────────────────────────────────────────
# Compute layer
# ─────────────────────────────────────────────────────────────────────────────

COMPUTE_NODES = [
    ComputeHost("HOST-DC1-01","Compute Host DC1 01","Dell","PowerEdge R750","Mumbai-DC1","Mumbai","West","10.50.1.1",64,512,"DC1-ROW-A-R01"),
    ComputeHost("HOST-DC1-02","Compute Host DC1 02","Dell","PowerEdge R750","Mumbai-DC1","Mumbai","West","10.50.1.2",64,512,"DC1-ROW-A-R02"),
    ComputeHost("HOST-DC1-03","Compute Host DC1 03","HPE","ProLiant DL380","Mumbai-DC1","Mumbai","West","10.50.1.3",48,384,"DC1-ROW-A-R03"),
    ComputeHost("HOST-DC2-01","Compute Host DC2 01","Dell","PowerEdge R750","Chennai-DC1","Chennai","South","10.50.2.1",64,512,"DC2-ROW-A-R01"),
    ComputeHost("HOST-DC2-02","Compute Host DC2 02","HPE","ProLiant DL380","Chennai-DC1","Chennai","South","10.50.2.2",48,384,"DC2-ROW-A-R02"),
]

VNFS = [
    VNF("VNF-AMF-01","AMF Instance 01","AMF","Nokia","22.6","HOST-DC1-01","10.60.1.1","active"),
    VNF("VNF-SMF-01","SMF Instance 01","SMF","Nokia","22.6","HOST-DC1-01","10.60.1.2","active"),
    VNF("VNF-UPF-01","UPF Instance 01","UPF","Nokia","22.6","HOST-DC1-02","10.60.1.3","active"),
    VNF("VNF-UPF-02","UPF Instance 02","UPF","Nokia","22.6","HOST-DC1-03","10.60.1.4","active"),
    VNF("VNF-PCF-01","PCF Instance 01","PCF","Ericsson","21.Q4","HOST-DC2-01","10.60.2.1","active"),
    VNF("VNF-AUSF-01","AUSF Instance 01","AUSF","Ericsson","21.Q4","HOST-DC2-01","10.60.2.2","active"),
    VNF("VNF-NRF-01","NRF Instance 01","NRF","Nokia","22.6","HOST-DC2-02","10.60.2.3","active"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Cross-domain dependency map
# Used by the fault propagation engine to determine cascade effects
# ─────────────────────────────────────────────────────────────────────────────

# Which optical fiber spans connect which cities
FIBER_SPANS = [
    ("ROADM-MUM-01","ROADM-CHN-01","MUM-CHN-SPAN-1",1400),
    ("ROADM-MUM-02","ROADM-CHN-02","MUM-CHN-SPAN-2",1400),
    ("ROADM-MUM-01","ROADM-BLR-01","MUM-BLR-SPAN-1",1000),
    ("ROADM-CHN-01","ROADM-BLR-01","CHN-BLR-SPAN-1",350),
    ("ROADM-MUM-01","ROADM-DEL-01","MUM-DEL-SPAN-1",1400),
    ("ROADM-MUM-01","ROADM-MUM-02","MUM-INTRA-01",30),
    ("ROADM-CHN-01","ROADM-CHN-02","CHN-INTRA-01",20),
]

# Which optical circuits underlay which IP links
OPTICAL_TO_IP = [
    ("ROADM-MUM-01","RTR-PE-MUM-01"),
    ("ROADM-MUM-02","RTR-PE-MUM-02"),
    ("ROADM-CHN-01","RTR-PE-CHN-01"),
    ("ROADM-CHN-02","RTR-PE-CHN-02"),
    ("ROADM-BLR-01","RTR-PE-BLR-01"),
    ("ROADM-DEL-01","RTR-PE-DEL-01"),
    ("ROADM-MUM-01","RTR-P-CORE-01"),
    ("ROADM-CHN-01","RTR-P-CORE-02"),
]

# Which PE router each gNB connects to for backhaul
GNB_BACKHAUL = {gnb.id: gnb.backhaul_pe for gnb in GNB_LIST}

# Which cells are affected when a PE router loses its optical circuit
# i.e. which gNBs backhaul via this PE
PE_TO_CELLS: Dict[str, List[int]] = {}
for gnb in GNB_LIST:
    pe = gnb.backhaul_pe
    if pe not in PE_TO_CELLS:
        PE_TO_CELLS[pe] = []
    PE_TO_CELLS[pe].extend([c.cell_index for c in gnb.cells])

# Which optical span failure affects which PE routers
SPAN_TO_PE: Dict[str, List[str]] = {
    "MUM-CHN-SPAN-1": ["RTR-PE-MUM-01","RTR-PE-CHN-01","RTR-P-CORE-01"],
    "MUM-CHN-SPAN-2": ["RTR-PE-MUM-02","RTR-PE-CHN-02"],
    "MUM-BLR-SPAN-1": ["RTR-PE-BLR-01"],
    "CHN-BLR-SPAN-1": ["RTR-PE-CHN-01","RTR-PE-BLR-01"],
    "MUM-DEL-SPAN-1": ["RTR-PE-DEL-01","RTR-P-CORE-03"],
    "MUM-INTRA-01":   ["RTR-PE-MUM-01","RTR-PE-MUM-02"],
    "CHN-INTRA-01":   ["RTR-PE-CHN-01","RTR-PE-CHN-02"],
}

# Which cells are affected by a fiber span cut (full cascade)
def cells_affected_by_span(span_id: str) -> List[int]:
    """Return list of cell_indices whose backhaul is lost when span_id is cut."""
    affected_cells = []
    for pe in SPAN_TO_PE.get(span_id, []):
        affected_cells.extend(PE_TO_CELLS.get(pe, []))
    return list(set(affected_cells))

# ─────────────────────────────────────────────────────────────────────────────
# Adjacency matrix for SIMBA
# ─────────────────────────────────────────────────────────────────────────────

def build_ran_adjacency() -> np.ndarray:
    """
    Build (21,21) adjacency matrix for SIMBA based on:
    1. Intra-gNB: cells sharing the same gNB are always connected
    2. Inter-gNB: cells on geographically adjacent gNBs are connected
    """
    adj = np.zeros((N_CELLS, N_CELLS), dtype=np.float32)

    # City proximity map — gNBs in same or adjacent city are inter-connected
    same_city_gnbs: Dict[str, List[str]] = {}
    for gnb in GNB_LIST:
        same_city_gnbs.setdefault(gnb.city, []).append(gnb.id)

    for i_cell in ALL_CELLS:
        for j_cell in ALL_CELLS:
            if i_cell.cell_index == j_cell.cell_index:
                continue
            i_gnb = GNB_BY_ID[i_cell.gnb_id]
            j_gnb = GNB_BY_ID[j_cell.gnb_id]
            # Intra-gNB
            if i_cell.gnb_id == j_cell.gnb_id:
                adj[i_cell.cell_index, j_cell.cell_index] = 1.0
            # Inter-gNB same city
            elif i_gnb.city == j_gnb.city:
                adj[i_cell.cell_index, j_cell.cell_index] = 1.0

    return adj

# ─────────────────────────────────────────────────────────────────────────────
# KPI schema — shared by SIMBA pipeline and fault propagation engine
# ─────────────────────────────────────────────────────────────────────────────

KPI_NAMES = [
    "rsrp_dbm", "rsrq_db", "sinr_db",
    "dl_throughput_mbps", "ul_throughput_mbps",
    "dl_bler_pct", "ul_bler_pct",
    "connected_ues", "handover_rate"
]
N_KPIS = len(KPI_NAMES)  # 9
