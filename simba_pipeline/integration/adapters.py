"""
Integration Adapters — Brownfield Telecom OSS/NMS Connectivity
===============================================================
Connects the SIMBA inference engine to existing telecom systems.

Design recommendation:
  - Use Kafka as the primary streaming backbone (not MCP)
  - MCP is appropriate for LLM-powered NOC assistant tools
  - Direct REST/SNMP/NETCONF adapters for topology discovery
  - TMF639 Resource Inventory API for CMDB enrichment

See module docstrings for detailed rationale.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime

from data.dataset_generator import KPI_NAMES, N_KPIS


# ─────────────────────────────────────────────────────────────────────────────
# Abstract KPI Source interface
# ─────────────────────────────────────────────────────────────────────────────

class KPISourceAdapter(ABC):
    """
    Base class for all KPI ingestion adapters.
    Each adapter connects to one NMS/EMS and emits
    (cell_id, kpi_vector) tuples at each polling interval.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the source system."""
        ...

    @abstractmethod
    def poll(self) -> Optional[np.ndarray]:
        """
        Fetch one KPI snapshot for all cells.
        Returns (n_cells, n_kpis) array or None if no data.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Kafka KPI Consumer
# ─────────────────────────────────────────────────────────────────────────────

class KafkaKPIAdapter(KPISourceAdapter):
    """
    Consumes KPI streams from a Kafka topic.

    This is the recommended integration pattern for brownfield
    telecom networks. The NMS/EMS publishes PM (Performance Management)
    data to Kafka topics. This adapter subscribes and assembles
    per-cell KPI vectors.

    Expected Kafka message schema (JSON):
    {
        "timestamp": "2024-01-15T14:23:01Z",
        "cell_id":   7,
        "gnb_id":    2,
        "kpis": {
            "rsrp_dbm":           -82.3,
            "rsrq_db":            -11.2,
            "sinr_db":             14.5,
            "dl_throughput_mbps":  75.2,
            "ul_throughput_mbps":  18.4,
            "dl_bler_pct":          2.1,
            "ul_bler_pct":          1.8,
            "connected_ues":       14.0,
            "handover_rate":        0.4
        }
    }

    Installation:
        pip install kafka-python
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic:             str,
        n_cells:           int,
        group_id:          str = "simba-consumer",
        auto_offset_reset: str = "latest",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic             = topic
        self.n_cells           = n_cells
        self.group_id          = group_id
        self.auto_offset_reset = auto_offset_reset
        self._consumer         = None
        self._cell_buffer:     Dict[int, np.ndarray] = {}

    def connect(self) -> None:
        try:
            from kafka import KafkaConsumer
            self._consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                consumer_timeout_ms=1000,
            )
            print(f"Kafka consumer connected to {self.bootstrap_servers} / {self.topic}")
        except ImportError:
            print("kafka-python not installed. Run: pip install kafka-python")
            raise

    def poll(self) -> Optional[np.ndarray]:
        """
        Reads messages until all cells have reported, then returns snapshot.
        In production this would use a more sophisticated time-bucketing approach.
        """
        if self._consumer is None:
            raise RuntimeError("Not connected. Call connect() first.")

        for message in self._consumer:
            payload = message.value
            cell_id = payload.get("cell_id")
            kpis    = payload.get("kpis", {})

            # Build KPI vector in canonical order
            kpi_vector = np.array(
                [kpis.get(k, 0.0) for k in KPI_NAMES],
                dtype=np.float32
            )
            self._cell_buffer[cell_id] = kpi_vector

            # If all cells have reported, return snapshot
            if len(self._cell_buffer) == self.n_cells:
                snapshot = np.array([
                    self._cell_buffer[i] for i in range(self.n_cells)
                ])
                self._cell_buffer.clear()
                return snapshot

        return None

    def disconnect(self) -> None:
        if self._consumer:
            self._consumer.close()
            self._consumer = None


# ─────────────────────────────────────────────────────────────────────────────
# REST PM Collector (Nokia NetAct, Ericsson ENM style)
# ─────────────────────────────────────────────────────────────────────────────

class RESTKPIAdapter(KPISourceAdapter):
    """
    Polls KPIs from a REST-based PM collector (e.g., Nokia NetAct,
    Ericsson ENM, Huawei iManager northbound REST API).

    This adapter uses a polling interval (typically 15 seconds to
    match standard PM granularity periods) to fetch aggregated KPIs.

    Installation:
        pip install requests
    """

    def __init__(
        self,
        base_url:      str,
        auth_token:    str,
        n_cells:       int,
        pm_endpoint:   str = "/pm/data/cells",
        poll_interval_s: float = 15.0,
    ):
        self.base_url         = base_url.rstrip("/")
        self.auth_token       = auth_token
        self.n_cells          = n_cells
        self.pm_endpoint      = pm_endpoint
        self.poll_interval_s  = poll_interval_s
        self._session         = None
        self._last_poll_time  = 0.0

    def connect(self) -> None:
        try:
            import requests
            self._session = requests.Session()
            self._session.headers.update({
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type":  "application/json",
                "Accept":        "application/json",
            })
            print(f"REST PM adapter connected to {self.base_url}")
        except ImportError:
            print("requests not installed. Run: pip install requests")
            raise

    def poll(self) -> Optional[np.ndarray]:
        """Fetch current KPIs from NMS REST API."""
        now = time.time()
        if now - self._last_poll_time < self.poll_interval_s:
            return None  # Not time to poll yet

        self._last_poll_time = now
        url = f"{self.base_url}{self.pm_endpoint}"
        try:
            resp = self._session.get(url, timeout=10)
            resp.raise_for_status()
            cells_data = resp.json()  # Expected: list of cell KPI dicts
            return self._parse_response(cells_data)
        except Exception as e:
            print(f"REST poll failed: {e}")
            return None

    def _parse_response(self, cells_data: List[Dict]) -> np.ndarray:
        """Parse NMS REST response into (n_cells, n_kpis) array."""
        snapshot = np.zeros((self.n_cells, N_KPIS), dtype=np.float32)
        for cell in cells_data:
            cell_id = cell.get("cellId", -1)
            if 0 <= cell_id < self.n_cells:
                kpis = cell.get("kpis", {})
                snapshot[cell_id] = [kpis.get(k, 0.0) for k in KPI_NAMES]
        return snapshot

    def disconnect(self) -> None:
        if self._session:
            self._session.close()
            self._session = None


# ─────────────────────────────────────────────────────────────────────────────
# Topology Discovery — NETCONF/RESTCONF
# ─────────────────────────────────────────────────────────────────────────────

class TopologyDiscoveryAdapter:
    """
    Discovers network topology from existing OSS systems.

    Strategy depends on what the brownfield network has available:

    Option 1 — NETCONF/RESTCONF (preferred for 5G):
        Query gNBs directly via YANG model (3GPP NRM)
        Endpoint: ietf-interfaces, 3gpp-nr-nrm-gnbdufunction

    Option 2 — TMF639 Resource Inventory API (standardised):
        Query the operator's CMDB/inventory system
        Returns topology as JSON-LD with standard resource types

    Option 3 — O-RAN SMO Discovery (for Open RAN):
        Query the Service Management & Orchestration framework
        Uses O-RAN WG3 E2 interface topology data

    Option 4 — Passive discovery via SNMP MIB walk:
        Walk IF-MIB and ENTITY-MIB to discover NEs and interfaces
        Build topology graph from CDP/LLDP neighbour tables

    Installation:
        pip install ncclient requests
    """

    def __init__(
        self,
        discovery_method: str = "tmf639",  # "netconf", "tmf639", "oran_smo", "snmp"
        endpoint:         str = "",
        credentials:      Dict[str, str] = None,
    ):
        self.discovery_method = discovery_method
        self.endpoint         = endpoint
        self.credentials      = credentials or {}

    def discover(self) -> Dict[str, Any]:
        """
        Run topology discovery.

        Returns:
            topology : dict with keys:
                cells      : list of {cell_id, gnb_id, site, lat, lon}
                adjacency  : np.ndarray (n_cells, n_cells)
                gnb_map    : dict cell_id -> gnb_id
        """
        if self.discovery_method == "tmf639":
            return self._discover_tmf639()
        elif self.discovery_method == "netconf":
            return self._discover_netconf()
        elif self.discovery_method == "oran_smo":
            return self._discover_oran_smo()
        else:
            raise ValueError(f"Unknown discovery method: {self.discovery_method}")

    def _discover_tmf639(self) -> Dict[str, Any]:
        """
        Discover topology via TMF639 Resource Inventory API.

        Example query:
          GET /resourceInventoryManagement/v4/resource
              ?resourceSpecification.name=NRCellDU
              &fields=id,name,resourceCharacteristic,place

        Returns list of NR cells with their characteristics
        (gnbId, cellLocalId, latitude, longitude, neighbors).
        """
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.credentials.get('token', '')}",
                "Accept": "application/json",
            }
            url = f"{self.endpoint}/resourceInventoryManagement/v4/resource"
            params = {"resourceSpecification.name": "NRCellDU",
                      "fields": "id,name,resourceCharacteristic,place"}
            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            resources = resp.json()
            return self._build_topology_from_tmf639(resources)
        except Exception as e:
            print(f"TMF639 discovery failed: {e}")
            return self._fallback_topology()

    def _discover_netconf(self) -> Dict[str, Any]:
        """
        Discover gNB topology via NETCONF using 3GPP NRM YANG models.

        Uses ncclient to connect to each gNB's NETCONF server
        and retrieve the NRCellDU inventory.
        """
        try:
            from ncclient import manager
            topology_cells = []
            for gnb_config in self.credentials.get("gnbs", []):
                with manager.connect(
                    host=gnb_config["host"],
                    port=gnb_config.get("port", 830),
                    username=gnb_config["username"],
                    password=gnb_config["password"],
                    hostkey_verify=False,
                ) as m:
                    # Get NR cell configuration
                    filter_xml = """
                    <filter>
                      <ManagedElement xmlns="urn:3gpp:sa5:_3gpp-common-managed-element">
                        <GNBDUFunction>
                          <NRCellDU/>
                        </GNBDUFunction>
                      </ManagedElement>
                    </filter>
                    """
                    result = m.get_config(source="running", filter=filter_xml)
                    topology_cells.extend(
                        self._parse_netconf_nrcelldu(result.xml, gnb_config)
                    )
            return self._build_topology_from_cells(topology_cells)
        except ImportError:
            print("ncclient not installed. Run: pip install ncclient")
            return self._fallback_topology()

    def _discover_oran_smo(self) -> Dict[str, Any]:
        """
        Discover topology from O-RAN Service Management & Orchestration.
        Uses O-RAN WG3 E2 interface topology data exposed by SMO northbound.
        """
        try:
            import requests
            url = f"{self.endpoint}/oran-smo/v1/topology/cells"
            resp = requests.get(url, timeout=15,
                               headers={"Authorization": f"Bearer {self.credentials.get('token','')}"})
            resp.raise_for_status()
            return self._build_topology_from_oran(resp.json())
        except Exception as e:
            print(f"O-RAN SMO discovery failed: {e}")
            return self._fallback_topology()

    def _build_topology_from_tmf639(self, resources: List[Dict]) -> Dict[str, Any]:
        """Parse TMF639 resource list into topology dict."""
        cells = []
        for r in resources:
            chars = {c["name"]: c.get("value") for c in r.get("resourceCharacteristic", [])}
            cells.append({
                "cell_id": len(cells),
                "tmf_id":  r.get("id"),
                "name":    r.get("name"),
                "gnb_id":  int(chars.get("gnbId", 0)),
                "lat":     float(chars.get("latitude",  0.0)),
                "lon":     float(chars.get("longitude", 0.0)),
            })
        return self._build_topology_from_cells(cells)

    def _build_topology_from_cells(self, cells: List[Dict]) -> Dict[str, Any]:
        """Build adjacency matrix from cell list using geographic proximity."""
        n = len(cells)
        adj = np.zeros((n, n), dtype=np.float32)
        gnb_map = {}
        for c in cells:
            gnb_map[c["cell_id"]] = c["gnb_id"]

        # Connect cells from same gNB (intra-site)
        for i in range(n):
            for j in range(n):
                if i != j and cells[i]["gnb_id"] == cells[j]["gnb_id"]:
                    adj[i, j] = 1.0

        # Connect cells from geographically adjacent gNBs
        # (using lat/lon distance if available, else default grid)
        for i in range(n):
            for j in range(n):
                if i != j and adj[i, j] == 0:
                    lat_diff = abs(cells[i].get("lat", 0) - cells[j].get("lat", 0))
                    lon_diff = abs(cells[i].get("lon", 0) - cells[j].get("lon", 0))
                    dist_deg = (lat_diff**2 + lon_diff**2)**0.5
                    if dist_deg < 0.005:  # roughly 500m
                        adj[i, j] = 1.0

        return {"cells": cells, "adjacency": adj, "gnb_map": gnb_map}

    @staticmethod
    def _fallback_topology(n_cells: int = 21) -> Dict[str, Any]:
        """
        Return a default hexagonal topology when discovery fails.
        Used as a safe fallback so the system degrades gracefully.
        """
        from data.dataset_generator import build_hexagonal_topology, build_adjacency_matrix
        cells_cfg = build_hexagonal_topology(n_sites=7)
        adj = build_adjacency_matrix(cells_cfg)
        cells = [{"cell_id": c.cell_id, "gnb_id": c.gnb_id,
                  "lat": 0.0, "lon": 0.0} for c in cells_cfg]
        gnb_map = {c.cell_id: c.gnb_id for c in cells_cfg}
        return {"cells": cells, "adjacency": adj, "gnb_map": gnb_map}

    def _parse_netconf_nrcelldu(self, xml_str: str, gnb_config: Dict) -> List[Dict]:
        """Parse NETCONF get-config response for NRCellDU elements."""
        import xml.etree.ElementTree as ET
        cells = []
        try:
            root = ET.fromstring(xml_str)
            ns = "urn:3gpp:sa5:_3gpp-common-managed-element"
            for cell_elem in root.iter(f"{{{ns}}}NRCellDU"):
                cell_local_id = cell_elem.find(f"{{{ns}}}cellLocalId")
                cells.append({
                    "cell_id": len(cells),
                    "gnb_id":  gnb_config.get("gnb_id", 0),
                    "name":    cell_elem.findtext(f"{{{ns}}}id", ""),
                    "lat":     gnb_config.get("lat", 0.0),
                    "lon":     gnb_config.get("lon", 0.0),
                })
        except Exception as e:
            print(f"NETCONF XML parse error: {e}")
        return cells

    def _build_topology_from_oran(self, data: Dict) -> Dict[str, Any]:
        cells = [
            {"cell_id": i, "gnb_id": c.get("gnbId", i // 3),
             "lat": c.get("latitude", 0.0), "lon": c.get("longitude", 0.0)}
            for i, c in enumerate(data.get("cells", []))
        ]
        return self._build_topology_from_cells(cells)


# ─────────────────────────────────────────────────────────────────────────────
# Integration Architecture Recommendation
# ─────────────────────────────────────────────────────────────────────────────

INTEGRATION_RECOMMENDATION = """
INTEGRATION ARCHITECTURE RECOMMENDATION
=========================================

Question: Should we use MCP (Model Context Protocol) for brownfield
telecom OSS/NMS integration?

SHORT ANSWER: No — MCP is the wrong tool for KPI stream ingestion.
Use it only for the LLM-powered NOC assistant layer.

DETAILED ANALYSIS:

1. MCP is designed for LLM context injection, not streaming data pipelines.
   MCP follows a request/response pattern (JSON-RPC 2.0). It has no native
   support for continuous data streams, back-pressure, or at-least-once
   delivery semantics. These are fundamental requirements for KPI streaming.

2. Telecom KPI data volumes are too high for MCP.
   A 1000-cell network at 1-second granularity produces 1000 messages/second.
   MCP has no buffering, partitioning, or consumer group semantics to handle this.

3. The right tool for each layer:

   ┌─────────────────────────────────────────────────────────────────┐
   │ LAYER              │ RECOMMENDED APPROACH                       │
   ├─────────────────────────────────────────────────────────────────┤
   │ KPI Stream Ingest  │ Apache Kafka (partitioned by gNB/domain)   │
   │                    │ → Handles high throughput, replay, exactly  │
   │                    │   once semantics                            │
   ├─────────────────────────────────────────────────────────────────┤
   │ Topology Discovery │ NETCONF/RESTCONF + YANG (direct to NE)      │
   │                    │ OR TMF639 Resource Inventory API            │
   │                    │ OR O-RAN SMO northbound API                 │
   │                    │ → One-time or periodic pull, not streaming  │
   ├─────────────────────────────────────────────────────────────────┤
   │ Alarm Ingest       │ TMF642 normalisation pipeline (already built)│
   │                    │ → SNMP → syslog → REST webhook → Kafka     │
   ├─────────────────────────────────────────────────────────────────┤
   │ SIMBA ML Inference │ Python service consuming from Kafka         │
   │                    │ → Outputs detections to Kafka output topic  │
   ├─────────────────────────────────────────────────────────────────┤
   │ NOC Assistant LLM  │ MCP IS APPROPRIATE HERE                     │
   │                    │ → Expose Kafka topics, Neo4j graph, SIMBA   │
   │                    │   predictions as MCP Resources/Tools        │
   │                    │ → LLM agent queries graph, asks "why is     │
   │                    │   cell 7 showing interference?"             │
   └─────────────────────────────────────────────────────────────────┘

4. MCP + Kafka complement each other:
   The Kafka streaming pipeline handles real-time data movement.
   MCP wraps the outputs (Neo4j KG, SIMBA predictions, TMF642 alarms)
   as tools that an LLM NOC assistant can query on demand.
   They are not competing — they serve different layers.

5. For topology discovery specifically:
   NETCONF/RESTCONF is the standard for modern 5G (3GPP NRM YANG models).
   TMF639 works well for operator CMDBs that already have TM Forum APIs.
   O-RAN SMO is the right approach for Open RAN deployments.
   Passive SNMP discovery (LLDP/CDP tables) works for legacy brownfield
   where no modern API exists.

RECOMMENDED BROWNFIELD INTEGRATION SEQUENCE:
  Step 1: Run TopologyDiscoveryAdapter once to pull cell/gNB topology
  Step 2: Store topology in Neo4j (build_graph.py already does this)
  Step 3: Subscribe to Kafka KPI topic (or deploy REST poll adapter)
  Step 4: Feed KPIs to SimbaInferenceEngine sliding window buffer
  Step 5: Output detections to Kafka output topic + Neo4j Incident nodes
  Step 6: Build MCP server exposing Neo4j + SIMBA outputs to NOC LLM
"""

if __name__ == "__main__":
    print(INTEGRATION_RECOMMENDATION)
