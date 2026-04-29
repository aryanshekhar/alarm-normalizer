"""
Data module — synthetic 5G KPI generation and preprocessing.

Key exports:
    KPITimeSeriesGenerator  — generates per-cell KPI time series with fault injection
    KPINormalizer           — min-max normaliser fitted on training data
    build_hexagonal_topology — creates 3GPP eMBB-Urban hexagonal cell layout
    build_adjacency_matrix  — builds cell adjacency from physical proximity
    create_sliding_windows  — converts time series to windowed ML input
    train_val_test_split    — temporal 50/25/25 split (no shuffle)
    KPI_NAMES               — ordered list of 9 KPI names
    N_KPIS                  — number of KPIs (9)
"""
from data.dataset_generator import (
    KPITimeSeriesGenerator,
    KPINormalizer,
    build_hexagonal_topology,
    build_adjacency_matrix,
    create_sliding_windows,
    train_val_test_split,
    KPI_NAMES,
    N_KPIS,
)
