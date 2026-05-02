from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from data_processing.constants import NUM_EDGE_TYPES
from .shared_encoder import SharedEncoder, EMB_DIM
from .base import BasePhase1Model

class DeletionLineRanker(nn.Module):
    """
    RankNet-style linear scoring head.
    """

    def __init__(self, hidden_dim: int = 1536):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )
        self.output = nn.Sigmoid()

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Pairwise forward for training.

        Args:
            emb1 : [B, hidden_dim]
            emb2 : [B, hidden_dim]
        Returns:
            prob : [B]  sigmoid(score(emb1) - score(emb2))
        """
        return self.output(self.model(emb1) - self.model(emb2)).squeeze(-1)

    def score(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Raw scalar score for a single embedding (inference).

        Args:
            emb : [hidden_dim] or [1, hidden_dim]
        Returns:
            scalar tensor
        """
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        return self.model(emb).squeeze()


class DeletionLineRankingModel(BasePhase1Model):
    """
    Full Phase 1 model: SharedEncoder + DeletionLineRanker.

    When include_unix=True  (default) PyG data carries token_ids +
    attention_mask; UnixCoder is fine-tuned jointly.
    When include_unix=False PyG data carries pre-computed x embeddings.

    predict() and forward() are inherited from BasePhase1Model.
    """

    def __init__(self, input_dim: int = EMB_DIM, 
                hidden_dim: int = 1536,
                num_gt_layers: int = 4, 
                num_heads: int = 8,
                num_edge_types: int = NUM_EDGE_TYPES,
                dropout: float = 0.2,
                include_unix: bool = True,
                num_unix_layers_freeze: int = 8,
                unix_chunk: int = 256,
                use_temporal_pe: bool = False):

        super().__init__()
        self.hidden_dim = hidden_dim
        self.include_unix = include_unix

        self.encoder = SharedEncoder(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            num_gt_layers=num_gt_layers, 
            num_heads=num_heads,
            num_edge_types=num_edge_types, 
            dropout=dropout,
            include_unix=include_unix,
            num_unix_layers_freeze=num_unix_layers_freeze,
            unix_chunk=unix_chunk,
            use_temporal_pe=use_temporal_pe,
        )
        self.ranker = DeletionLineRanker(hidden_dim)