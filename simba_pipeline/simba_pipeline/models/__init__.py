"""
Models module — SIMBA neural network architecture.

Key exports:
    Simba                   — full GNN+Transformer model
    GraphStructureLearning  — learns soft adjacency matrix from node embeddings
    GraphConvolutionModule  — stacked GCN layers for spatial feature extraction
    TransformerBranch       — multi-head attention for temporal modelling
    WeightedFocalLoss       — imbalance-aware focal loss for training
    compute_class_weights   — inverse-frequency class weight computation
"""
from models.simba import (
    Simba,
    GraphStructureLearning,
    GraphConvolutionModule,
    TransformerBranch,
    WeightedFocalLoss,
    compute_class_weights,
)
