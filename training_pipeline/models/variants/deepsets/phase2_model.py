import torch
import torch.nn as nn


class DeepSetsCommitRankingModule(nn.Module):

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project Phase 1 output dim → Phase 2 hidden dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ρ: post-pooling MLP — scores each commit independently
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        commit_indices: torch.Tensor,
        is_temporal_node: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            node_embeddings  : [N, input_dim]  pre-computed by frozen encoder
            commit_indices   : [N]             which commit each node belongs to
            is_temporal_node : [N]             IGNORED

        Returns:
            scores : [C]  per-commit ranking scores
        """
        device = node_embeddings.device
        N = node_embeddings.size(0)
        num_commits = int(commit_indices.max().item()) + 1

        # Project: [N, input_dim] → [N, hidden_dim]
        h = self.input_proj(node_embeddings)

        # Σ: mean pool per commit — order-agnostic aggregation
        h_sum = torch.zeros(num_commits, self.hidden_dim, device=device)
        counts = torch.zeros(num_commits, 1, device=device)

        idx = commit_indices.unsqueeze(1).expand_as(h)
        h_sum.scatter_add_(0, idx, h)
        counts.scatter_add_(
            0,
            commit_indices.unsqueeze(1),
            torch.ones(N, 1, device=device),
        )

        commit_embeddings = h_sum / counts.clamp(min=1.0)  # [C, hidden_dim]

        # ρ: score each commit independently
        return self.rho(commit_embeddings).squeeze(-1)  # [C]