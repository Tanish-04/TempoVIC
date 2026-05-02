"""
Section Transformer encoder variant — standard nn.TransformerEncoder per section.
Inherits UnixCoder logic and encode_pyg() from BaseEncoder.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.base import BaseEncoder
from models.shared_encoder import EMB_DIM


class SectionTransformerEncoder(BaseEncoder):

    def __init__(
        self,
        input_dim: int = EMB_DIM,
        hidden_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        include_unix: bool = True,
        num_unix_layers_freeze: int = 0,
        unix_chunk: int = 256,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._init_unix(include_unix, num_unix_layers_freeze, unix_chunk, use_checkpoint)

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Standard Transformer encoder — no edge types, no graph structure.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def _run_per_section(
        self,
        x: torch.Tensor,
        temporal_pos: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        device = x.device
        N = x.size(0)
        h_out = torch.zeros(N, self.hidden_dim, device=device)

        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=device)

        # Composite section key: unique (graph, temporal_pos) pair
        max_tp = int(temporal_pos.max().item()) + 1
        composite = batch * max_tp + temporal_pos
        unique_sections = composite.unique()

        for sec in unique_sections:
            mask = composite == sec
            indices = mask.nonzero(as_tuple=True)[0]
            section_x = x[indices]

            section_h = self.input_proj(section_x)

            section_h = section_h.unsqueeze(1)
            section_h = self.transformer(section_h)
            section_h = section_h.squeeze(1)

            h_out[indices] = section_h

        return h_out

    def forward(
        self,
        x: torch.Tensor = None,
        edge_index: torch.Tensor = None,   
        edge_type: torch.Tensor = None,   
        temporal_pos: torch.Tensor = None,
        token_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.include_unix and token_ids is not None:
            x = self._run_unix(token_ids, attention_mask)

        if temporal_pos is not None:
            h = self._run_per_section(x, temporal_pos, batch)
        else:
            h = self.input_proj(x)
            h = self.transformer(h.unsqueeze(1)).squeeze(1)

        return h