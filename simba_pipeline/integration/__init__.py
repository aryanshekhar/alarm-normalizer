"""
Integration module — brownfield OSS/NMS connectivity adapters.

Key exports:
    KafkaKPIAdapter         — Kafka consumer for streaming KPI ingestion
    RESTKPIAdapter          — REST polling adapter for NMS PM APIs
    TopologyDiscoveryAdapter — NETCONF/TMF639/O-RAN SMO topology discovery
    INTEGRATION_RECOMMENDATION — architecture decision guide (MCP vs Kafka)
"""
from integration.adapters import (
    KafkaKPIAdapter,
    RESTKPIAdapter,
    TopologyDiscoveryAdapter,
    INTEGRATION_RECOMMENDATION,
)
