from models.phase1_model import DeletionLineRanker
from models.shared_encoder import EMB_DIM
from models.base import BasePhase1Model
from models.variants.deepsets.encoder import DeepSetsEncoder


class DeepSetsDeletionLineRankingModel(BasePhase1Model):
    """
    Phase 1 deletion-line ranker using DeepSets order-agnostic encoding.
    """

    def __init__(
        self,
        input_dim: int = EMB_DIM,
        hidden_dim: int = 768,
        dropout: float = 0.1,
        include_unix: bool = True,
        num_unix_layers_freeze: int = 8,
        unix_chunk: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.include_unix = include_unix

        self.encoder = DeepSetsEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            include_unix=include_unix,
            num_unix_layers_freeze=num_unix_layers_freeze,
            unix_chunk=unix_chunk,
        )
        self.ranker = DeletionLineRanker(hidden_dim)