import torch
import torch.nn as nn


class TransformerCommitRankingModule(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Single shared query — no dual query in this ablation since
        # is_temporal_node is not meaningful without temporal edges
        self.shared_query = nn.Parameter(torch.randn(num_heads, self.head_dim))

        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj_pool = nn.Linear(hidden_dim, hidden_dim)
        self.norm_pool = nn.LayerNorm(hidden_dim)

        self.ranking_head = nn.Sequential(
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
        device = node_embeddings.device
        N = node_embeddings.size(0)
        num_commits = commit_indices.max().item() + 1

        # Project: [N, input_dim] → [N, hidden_dim]
        node_embeddings = self.input_proj(node_embeddings)

        # Keys and values
        K_all = self.k_proj(node_embeddings).view(N, self.num_heads, self.head_dim)
        V_all = self.v_proj(node_embeddings).view(N, self.num_heads, self.head_dim)

        # Shared query for all nodes
        queries = self.shared_query.unsqueeze(0).expand(N, -1, -1) 

        # Attention logits
        logits = (queries * K_all).sum(dim=-1) * self.scale  

        idx = commit_indices.unsqueeze(1).expand_as(logits) 

        logits_stable = logits - logits.max().detach()
        exp_logits = torch.exp(logits_stable) 

        # Per-commit softmax
        exp_sum = torch.zeros(num_commits, self.num_heads, device=device)
        exp_sum.scatter_add_(0, idx, exp_logits)
        attn_weights = exp_logits / (exp_sum[commit_indices] + 1e-9) 

        # Weighted sum of values per commit
        weighted_V = attn_weights.unsqueeze(-1) * V_all 
        idx_v = commit_indices.view(N, 1, 1).expand_as(weighted_V)

        pooled = torch.zeros(num_commits, self.num_heads, self.head_dim, device=device)
        pooled.scatter_add_(0, idx_v, weighted_V)

        commit_embeddings = self.norm_pool(
            self.out_proj_pool(pooled.reshape(num_commits, -1))
        ) 

        return self.ranking_head(commit_embeddings).squeeze(-1) 
