"""
Base classes for encoders and Phase 1 models.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from transformers import AutoModel

from models.shared_encoder import MODEL_NAME


class BaseEncoder(nn.Module):
    """
    Abstract base class for all Phase 1 encoders.
    """

    def _init_unix(
        self,
        include_unix: bool = True,
        num_unix_layers_freeze: int = 0,
        unix_chunk: int = 256,
        use_checkpoint: bool = True,
    ) -> None:
        """Initialise the UnixCoder language model and freeze bottom layers."""
        self.include_unix = include_unix
        self.unix_chunk = unix_chunk
        self.use_checkpoint = use_checkpoint

        if include_unix:
            self.unix_model = AutoModel.from_pretrained(MODEL_NAME)
            if num_unix_layers_freeze > 0:
                for p in self.unix_model.embeddings.parameters():
                    p.requires_grad = False
                for i in range(
                    min(num_unix_layers_freeze,
                        len(self.unix_model.encoder.layer))
                ):
                    for p in self.unix_model.encoder.layer[i].parameters():
                        p.requires_grad = False

    def _run_unix(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run UnixCoder in chunks with optional gradient checkpointing.
        """
        dev = next(self.unix_model.parameters()).device

        unix_anchor = next(
            (
                p
                for name, p in self.unix_model.named_parameters()
                if p.requires_grad and "pooler" not in name
            ),
            None,
        )
        can_checkpoint = self.use_checkpoint and unix_anchor is not None

        def _cls(anchor, ids, mask):
            return self.unix_model(
                input_ids=ids, attention_mask=mask
            ).last_hidden_state[:, 0, :]

        pieces = []
        for i in range(0, token_ids.size(0), self.unix_chunk):
            ids_c = token_ids[i : i + self.unix_chunk].to(dev)
            mask_c = attention_mask[i : i + self.unix_chunk].to(dev)
            if self.training and can_checkpoint:
                emb = torch.utils.checkpoint.checkpoint(
                    _cls, unix_anchor, ids_c, mask_c
                )
            else:
                emb = _cls(unix_anchor, ids_c, mask_c)
            pieces.append(emb)
        return torch.cat(pieces, dim=0)

    def encode_pyg(self, pyg_data) -> torch.Tensor:
        """
        Unpack a PyG Data object and run the encoder — single entry-point
        used by both Phase 1 and Phase 2 models so unpacking logic is
        not duplicated across model files.
        """
        dev = next(self.parameters()).device
        kw = {
            "edge_index": pyg_data.edge_index.to(dev),
            "edge_type":  pyg_data.edge_type.to(dev),
            "temporal_pos": (
                pyg_data.temporal_pos.to(dev)
                if hasattr(pyg_data, "temporal_pos")
                and pyg_data.temporal_pos is not None
                else None
            ),
            "batch": (
                pyg_data.batch.to(dev)
                if hasattr(pyg_data, "batch")
                and pyg_data.batch is not None
                else None
            ),
        }
        if self.include_unix and hasattr(pyg_data, "token_ids"):
            kw["token_ids"]      = pyg_data.token_ids.to(dev)
            kw["attention_mask"] = pyg_data.attention_mask.to(dev)
        else:
            kw["x"] = pyg_data.x.to(dev)
        return self.forward(**kw)



class BasePhase1Model(nn.Module):
    """
    Abstract base class for all Phase 1 deletion-line ranking models.
    """

    def predict(self, pyg_data, del_idx: int = 0) -> torch.Tensor:
        """
        Score a single deletion line for inference.

        Returns:
            scalar ranking score
        """
        if isinstance(del_idx, torch.Tensor):
            del_idx = del_idx.item() if del_idx.numel() == 1 else del_idx[0].item()
        h = self.encoder.encode_pyg(pyg_data)
        if del_idx >= h.size(0):
            raise ValueError(
                f"del_idx {del_idx} out of bounds for graph with {h.size(0)} nodes"
            )
        return self.ranker.score(h[del_idx])

    def forward(
        self,
        mini_graphs,
        pairs,
        device: torch.device,
        max_nodes: int = 9500,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], List[int]]:
        """
        Encode every graph in mini_graphs exactly once, then extract
        deletion-line embeddings for every pair by lookup.
        """
        # Step 1: filter oversized graphs, build id→embedding map
        valid_pygs:   List           = []
        valid_gids:   List[int]      = []
        skipped_gids: set            = set()

        seen_gids: set = set()
        for mg in mini_graphs:
            gid = id(mg.pyg)
            if gid in seen_gids:
                continue
            seen_gids.add(gid)

            if mg.pyg is None:
                skipped_gids.add(gid)
                continue
            if mg.pyg.num_nodes > max_nodes:
                skipped_gids.add(gid)
                continue

            valid_pygs.append(mg.pyg.to(device))
            valid_gids.append(gid)

        if not valid_pygs:
            return None, None, []

        # Step 2: single batched encoder forward pass
        batched = Batch.from_data_list(valid_pygs)
        h_all   = self.encoder.encode_pyg(batched)

        # Map graph id → deletion-line embedding (node 0 of each graph)
        gid_to_emb: Dict[int, torch.Tensor] = {}
        for i, gid in enumerate(valid_gids):
            node_start = batched.ptr[i].item()
            emb = h_all[node_start]
            gid_to_emb[gid] = emb if self.training else emb.detach()

        # Step 3: pair lookup — zero additional encoding
        emb_x_list:  List[torch.Tensor] = []
        emb_y_list:  List[torch.Tensor] = []
        valid_mask:  List[int]          = []

        for i, pair in enumerate(pairs):
            gx = id(pair.x.pyg)
            gy = id(pair.y.pyg)

            if gx in skipped_gids or gy in skipped_gids:
                continue
            if gx not in gid_to_emb or gy not in gid_to_emb:
                continue
            emb_x_list.append(gid_to_emb[gx])
            emb_y_list.append(gid_to_emb[gy])
            valid_mask.append(i)

        if not emb_x_list:
            return None, None, []

        emb_x = torch.stack(emb_x_list)
        emb_y = torch.stack(emb_y_list)

        return emb_x, emb_y, valid_mask
