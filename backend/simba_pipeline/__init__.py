"""
SIMBA Pipeline — GNN+Transformer RCA for 5G RAN
================================================
Based on: arXiv:2406.15638 (Hasan et al., 2024)

Package structure:
    data/            — KPI dataset generator and normaliser
    models/          — SIMBA model (GSL + GCN + Transformer)
    training/        — Training loop, metrics, early stopping
    inference/       — Real-time inference engine
    integration/     — Brownfield OSS/NMS adapters
"""
__version__ = "1.0.0"
__author__  = "Based on SIMBA by Hasan et al., arXiv:2406.15638"
