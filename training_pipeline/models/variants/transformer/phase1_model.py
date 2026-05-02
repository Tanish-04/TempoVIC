"""
Section Transformer Phase 1 ablation model.

Inherits predict() and forward() from BasePhase1Model.
Only __init__ differs — uses SectionTransformerEncoder instead of SharedEncoder.
"""

from models.phase1_model import DeletionLineRanker
from models.shared_encoder import EMB_DIM
from models.base import BasePhase1Model
from models.variants.transformer.encoder import SectionTransformerEncoder


class AblationDeletionLineRankingModel(BasePhase1Model):
    """
    Phase 1 model using SectionTransformerEncoder for clean no_temporal ablation.

    Architecture is identical to DeletionLineRankingModel:
        encoder → extract h[del_idx] → ranker → score

    Only the encoder is different (Transformer per section instead of GAT).
    """

    def __init__(
        self,
        input_dim: int = EMB_DIM,
        hidden_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        include_unix: bool = True,
        num_unix_layers_freeze: int = 8,
        unix_chunk: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.include_unix = include_unix

        self.encoder = SectionTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            include_unix=include_unix,
            num_unix_layers_freeze=num_unix_layers_freeze,
            unix_chunk=unix_chunk,
        )
        self.ranker = DeletionLineRanker(hidden_dim)