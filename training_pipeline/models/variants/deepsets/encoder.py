"""
DeepSets encoder variant — order-agnostic encoding via φ → mean pool → ρ.
Inherits UnixCoder logic and encode_pyg() from BaseEncoder.
"""

from typing import Optional

import torch
import torch.nn as nn

from models.base import BaseEncoder
from models.shared_encoder import EMB_DIM


class DeepSetsEncoder(BaseEncoder):
    """
    Order-agnostic encoder implementing the DeepSets decomposition,
    processed section-by-section for bounded memory.
    """

    def __init__(
        self,
        input_dim: int = EMB_DIM,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        include_unix: bool = True,
        num_unix_layers_freeze: int = 0,
        unix_chunk: int = 256,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self._init_unix(include_unix, num_unix_layers_freeze, unix_chunk, use_checkpoint)

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _run_deepsets_per_section(
        self,
        x: torch.Tensor,
        temporal_pos: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply φ → mean pool → ρ independently per (graph, temporal_pos)
        section, looping in Python so each iteration's activations are
        released before the next one starts.

        Memory per step is O(S_i * hidden_dim) where S_i is the largest
        section, not O(N_total * hidden_dim).

        Args:
            x            : [N, input_dim]  node embeddings from unix / x
            temporal_pos : [N]             section index per node
            batch        : [N]             graph index per node

        Returns:
            h_out : [N, hidden_dim]  each node replaced by ρ(mean(φ(section)))
        """
        device = x.device
        N = x.size(0)
        h_out = torch.zeros(N, self.hidden_dim, device=device)

        max_tp = int(temporal_pos.max().item()) + 1
        composite = batch * max_tp + temporal_pos
        unique_sections = composite.unique()

        for sec in unique_sections:
            mask = composite == sec
            indices = mask.nonzero(as_tuple=True)[0]
            section_x = x[indices]

            section_h = self.phi(section_x)
            pooled = section_h.mean(dim=0, keepdim=True)
            refined = self.rho(pooled)

            h_out[indices] = refined.expand(indices.size(0), -1)

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

        device = x.device

        if temporal_pos is not None:
            if batch is None:
                batch = torch.zeros(
                    x.size(0), dtype=torch.long, device=device
                )
            h = self._run_deepsets_per_section(x, temporal_pos, batch)
        else:
            section_h = self.phi(x)
            pooled = section_h.mean(dim=0, keepdim=True)
            refined = self.rho(pooled)
            h = refined.expand(x.size(0), -1)

        return h