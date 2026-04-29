"""
SIMBA Model Implementation
===========================
GNN + Transformer for anomaly detection and RCA in 5G RAN.

Architecture (following arXiv:2406.15638):

  Input: KPI time series (window_size, n_cells, n_kpis)
      ↓
  [1] Graph Structure Learning (GSL)
      Learns adjacency matrix from data, combining with physical topology prior
      ↓
  [2] Graph Convolution (GC)
      Captures spatial (inter-cell) dependencies using learned graph
      ↓
  [3] Transformer Branch
      Captures temporal dependencies across the window
      ↓
  [4] Fusion + Feed-Forward
      Combines spatial and temporal embeddings
      ↓
  Output: Class probabilities (normal, power_reduction, interference) per cell

References:
  - SIMBA: arXiv:2406.15638 (Hasan et al., 2024)
  - MTGNN: arXiv:2005.11650 (Wu et al., 2020) — spatial backbone
  - Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Graph Structure Learning Module
# ─────────────────────────────────────────────────────────────────────────────

class GraphStructureLearning(nn.Module):
    """
    Learns a soft adjacency matrix from node embeddings.
    Combines learned structure with a physical topology prior.

    Each cell is represented by a learnable embedding vector.
    The adjacency is computed as a scaled dot-product similarity.
    """

    def __init__(
        self,
        n_cells:    int,
        embed_dim:  int = 32,
        alpha:      float = 3.0,   # sharpness of softmax
        top_k:      int  = 5,      # keep top-k connections per node
        prior_weight: float = 0.5, # weight for physical topology prior
    ):
        super().__init__()
        self.n_cells      = n_cells
        self.alpha        = alpha
        self.top_k        = top_k
        self.prior_weight = prior_weight

        # Learnable node embeddings
        self.node_emb1 = nn.Embedding(n_cells, embed_dim)
        self.node_emb2 = nn.Embedding(n_cells, embed_dim)

        # Linear projections
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_emb1.weight)
        nn.init.xavier_uniform_(self.node_emb2.weight)

    def forward(
        self,
        prior: Optional[torch.Tensor] = None  # (n_cells, n_cells) physical adj
    ) -> torch.Tensor:
        """
        Returns:
            adj : (n_cells, n_cells) soft adjacency matrix, row-normalised
        """
        idx = torch.arange(self.n_cells, device=self.node_emb1.weight.device)

        # Compute pairwise similarity
        e1 = torch.tanh(self.alpha * self.proj1(self.node_emb1(idx)))  # (N, D)
        e2 = torch.tanh(self.alpha * self.proj2(self.node_emb2(idx)))  # (N, D)
        adj = torch.mm(e1, e2.T)  # (N, N)

        # Top-K sparsification — keep only top_k strongest connections
        if self.top_k < self.n_cells:
            topk_vals, _ = torch.topk(adj, self.top_k, dim=1)
            threshold     = topk_vals[:, -1:].expand_as(adj)
            adj           = torch.where(adj >= threshold, adj,
                                        torch.zeros_like(adj))

        # Optionally fuse with physical topology prior
        if prior is not None:
            adj = (1 - self.prior_weight) * adj + self.prior_weight * prior

        # Row-wise softmax normalisation (removes self-loops)
        adj = adj.fill_diagonal_(0)
        adj = F.softmax(adj, dim=1)
        adj = adj.fill_diagonal_(0)  # ensure no self-loops after softmax

        return adj


# ─────────────────────────────────────────────────────────────────────────────
# 2. Graph Convolution Module
# ─────────────────────────────────────────────────────────────────────────────

class GraphConvolutionLayer(nn.Module):
    """
    Single graph convolution layer.
    H' = σ(A * H * W) where A is the adjacency matrix.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : (batch, n_cells, in_features)
            adj : (n_cells, n_cells)
        Returns:
            out : (batch, n_cells, out_features)
        """
        support = torch.matmul(x, self.weight)         # (B, N, out)
        out     = torch.matmul(adj.unsqueeze(0), support)  # (B, N, out)
        if self.bias is not None:
            out = out + self.bias
        return F.relu(out)


class GraphConvolutionModule(nn.Module):
    """
    Stacked GCN layers with residual connections.
    Takes the full time series window, applies GCN at each timestep,
    then pools temporally to produce a spatial feature embedding.
    """

    def __init__(
        self,
        n_kpis:     int,
        n_cells:    int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        n_layers:   int = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.n_layers  = n_layers

        # Input projection
        self.input_proj = nn.Linear(n_kpis, hidden_dim)

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(n_layers):
            in_f  = hidden_dim
            out_f = output_dim if i == n_layers - 1 else hidden_dim
            self.gcn_layers.append(GraphConvolutionLayer(in_f, out_f))

        # Batch norm per layer
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(n_cells)
            for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Residual projection if dimensions change
        self.residual_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else None

    def forward(
        self,
        x:   torch.Tensor,  # (batch, window, n_cells, n_kpis)
        adj: torch.Tensor,  # (n_cells, n_cells)
    ) -> torch.Tensor:
        """
        Returns:
            spatial_emb : (batch, n_cells, output_dim)
        """
        B, T, N, K = x.shape

        # Project input KPIs to hidden dimension
        h = self.input_proj(x)  # (B, T, N, hidden_dim)

        # Apply GCN at each timestep, then average pool over time
        # We average pool first for efficiency, then apply GCN
        h_pooled = h.mean(dim=1)  # (B, N, hidden_dim)
        residual  = h_pooled

        for i, (gcn, bn) in enumerate(zip(self.gcn_layers, self.bns)):
            h_pooled = gcn(h_pooled, adj)  # (B, N, out)
            h_pooled = bn(h_pooled)
            h_pooled = self.dropout(h_pooled)

        # Add residual
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        spatial_emb = h_pooled + residual

        return spatial_emb  # (B, N, output_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transformer Branch
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBranch(nn.Module):
    """
    Transformer encoder for temporal dependency modelling.

    Operates on per-cell KPI sequences independently then
    aggregates across cells.

    Input:  (batch, window, n_cells, n_kpis)
    Output: (batch, n_cells, temporal_dim)
    """

    def __init__(
        self,
        n_kpis:       int,
        n_cells:      int,
        temporal_dim: int   = 64,
        n_heads:      int   = 4,
        n_layers:     int   = 2,
        ff_dim:       int   = 128,
        dropout:      float = 0.1,
        window_size:  int   = 30,
    ):
        super().__init__()
        self.n_cells      = n_cells
        self.temporal_dim = temporal_dim

        # Input projection: KPIs → temporal_dim
        self.input_proj = nn.Linear(n_kpis, temporal_dim)
        self.pos_enc    = PositionalEncoding(temporal_dim, max_len=window_size+10,
                                              dropout=dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(temporal_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch, window, n_cells, n_kpis)
        Returns:
            temporal_emb : (batch, n_cells, temporal_dim)
        """
        B, T, N, K = x.shape

        # Reshape to process each cell's time series independently
        # (B*N, T, K) — treat each cell as an independent sequence
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, K)

        # Project to temporal_dim and add positional encoding
        h = self.input_proj(x_flat)     # (B*N, T, temporal_dim)
        h = self.pos_enc(h)

        # Transformer encoding
        h = self.transformer(h)         # (B*N, T, temporal_dim)
        h = self.norm(h)

        # Pool over time dimension (CLS-style: use last timestep)
        h = h[:, -1, :]                 # (B*N, temporal_dim)

        # Reshape back to (B, N, temporal_dim)
        temporal_emb = h.reshape(B, N, self.temporal_dim)
        return temporal_emb


# ─────────────────────────────────────────────────────────────────────────────
# 4. SIMBA: Full model combining GSL + GCN + Transformer + Fusion
# ─────────────────────────────────────────────────────────────────────────────

class Simba(nn.Module):
    """
    SIMBA: Spatio-temporal anomaly detection and RCA for 5G RAN.

    Combines:
      - Graph Structure Learning (learns cell relationship graph from data)
      - Graph Convolution (spatial feature extraction)
      - Transformer (temporal feature extraction)
      - Fusion head (combines spatial+temporal, outputs per-cell class probs)

    Output per cell: probability distribution over
      [normal, excessive_power_reduction, interference]
    """

    N_CLASSES = 3

    def __init__(
        self,
        n_kpis:       int,
        n_cells:      int,
        window_size:  int   = 30,
        gsl_embed_dim: int  = 32,
        gsl_top_k:    int   = 5,
        gcn_hidden:   int   = 64,
        gcn_output:   int   = 64,
        gcn_layers:   int   = 2,
        temporal_dim: int   = 64,
        n_heads:      int   = 4,
        transformer_layers: int = 2,
        ff_dim:       int   = 128,
        fusion_hidden: int  = 128,
        dropout:      float = 0.1,
        prior_weight: float = 0.5,
    ):
        super().__init__()
        self.n_cells  = n_cells
        self.n_kpis   = n_kpis

        # Module 1: Graph Structure Learning
        self.gsl = GraphStructureLearning(
            n_cells=n_cells,
            embed_dim=gsl_embed_dim,
            top_k=gsl_top_k,
            prior_weight=prior_weight,
        )

        # Module 2: Graph Convolution
        self.gcn = GraphConvolutionModule(
            n_kpis=n_kpis,
            n_cells=n_cells,
            hidden_dim=gcn_hidden,
            output_dim=gcn_output,
            n_layers=gcn_layers,
            dropout=dropout,
        )

        # Module 3: Transformer
        self.transformer_branch = TransformerBranch(
            n_kpis=n_kpis,
            n_cells=n_cells,
            temporal_dim=temporal_dim,
            n_heads=n_heads,
            n_layers=transformer_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            window_size=window_size,
        )

        # Module 4: Fusion feed-forward head
        fused_dim = gcn_output + temporal_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, self.N_CLASSES),
        )

        self._init_fusion_weights()

    def _init_fusion_weights(self):
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x:     torch.Tensor,                  # (B, T, N, K)
        prior: Optional[torch.Tensor] = None, # (N, N) physical adjacency
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x     : (batch, window_size, n_cells, n_kpis) — normalised KPI window
            prior : (n_cells, n_cells) — physical topology adjacency (optional)

        Returns:
            logits : (batch, n_cells, n_classes) — raw logits
            adj    : (n_cells, n_cells) — learned adjacency (for visualisation)
        """
        # Step 1: Learn graph structure
        adj = self.gsl(prior)  # (N, N)

        # Step 2: Spatial embedding via GCN
        spatial_emb = self.gcn(x, adj)      # (B, N, gcn_output)

        # Step 3: Temporal embedding via Transformer
        temporal_emb = self.transformer_branch(x)  # (B, N, temporal_dim)

        # Step 4: Fuse and classify
        fused  = torch.cat([spatial_emb, temporal_emb], dim=-1)  # (B, N, fused)
        logits = self.fusion(fused)                               # (B, N, n_classes)

        return logits, adj

    def predict(
        self,
        x:     torch.Tensor,
        prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns class probabilities (softmax over logits)."""
        with torch.no_grad():
            logits, _ = self.forward(x, prior)
        return F.softmax(logits, dim=-1)  # (B, N, n_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Weighted Cross-Entropy Loss (handles class imbalance per SIMBA paper)
# ─────────────────────────────────────────────────────────────────────────────

class WeightedFocalLoss(nn.Module):
    """
    Focal loss variant of weighted cross-entropy.
    Focuses training on hard/misclassified examples.
    Particularly useful for imbalanced datasets (2% anomaly rate).

    L = -w_y * (1 - p_y)^gamma * log(p_y)
    """

    def __init__(
        self,
        n_classes:     int   = 3,
        class_weights: Optional[torch.Tensor] = None,
        gamma:         float = 2.0,   # focusing parameter
    ):
        super().__init__()
        self.gamma         = gamma
        self.class_weights = class_weights

    def forward(
        self,
        logits: torch.Tensor,  # (B, N, n_classes) or (B*N, n_classes)
        labels: torch.Tensor,  # (B, N) or (B*N,)
    ) -> torch.Tensor:
        # Flatten if needed
        if logits.dim() == 3:
            B, N, C = logits.shape
            logits = logits.reshape(B * N, C)
            labels = labels.reshape(B * N)

        log_p = F.log_softmax(logits, dim=1)
        p     = torch.exp(log_p)

        # Gather log_p for the true class
        log_p_true = log_p.gather(1, labels.unsqueeze(1)).squeeze(1)
        p_true     = p.gather(1, labels.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1.0 - p_true) ** self.gamma

        # Class weight
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            sample_weight = cw[labels]
        else:
            sample_weight = torch.ones_like(log_p_true)

        loss = -focal_weight * sample_weight * log_p_true
        return loss.mean()


def compute_class_weights(
    y_train: np.ndarray,
    n_classes: int = 3,
    method: str = "inverse_freq",
) -> torch.Tensor:
    """
    Compute class weights to handle imbalance.
    inverse_freq: w_c = total_samples / (n_classes * count_c)
    """
    counts = np.bincount(y_train.flatten(), minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    if method == "inverse_freq":
        weights = len(y_train.flatten()) / (n_classes * counts)
    else:
        weights = np.ones(n_classes)
    return torch.tensor(weights, dtype=torch.float32)
