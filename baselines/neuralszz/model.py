from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer, AutoModel
from torch_geometric.data import HeteroData
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from torch import nn

import torch_geometric.transforms as T
from torch_geometric.nn import HANConv

from torch_geometric.data import HeteroData
import json
import random


class HAN(nn.Module):
    def __init__(
        self,
        device,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata,
        heads=2,
        dropout=0.0,
        num_bert_layers_freeze: int = 0,
    ):
        super().__init__()
        self.device = device
        self.bert_model = AutoModel.from_pretrained("microsoft/unixcoder-base-nine")

        if num_bert_layers_freeze > 0:
            for p in self.bert_model.embeddings.parameters():
                p.requires_grad = False
            n_layers = len(self.bert_model.encoder.layer)
            for i in range(min(num_bert_layers_freeze, n_layers)):
                for p in self.bert_model.encoder.layer[i].parameters():
                    p.requires_grad = False
            print(f"  BERT: froze embeddings + bottom {min(num_bert_layers_freeze, n_layers)} "
                  f"of {n_layers} layers; fine-tuning top {n_layers - min(num_bert_layers_freeze, n_layers)}")

        self.han_conv = HANConv(in_channels, out_channels, metadata, heads, 1, dropout)

    def forward(self, pyg, delIndexes):
        token_ids_dict = pyg.token_ids_dict
        if token_ids_dict["add_node"].numel() != 0:
            pyg["add_node"].x = self.bert_model(
                torch.tensor(
                    token_ids_dict["add_node"].tolist(),
                    dtype=torch.long,
                    device=self.device,
                )
            )[0][:, 0, :]
        if token_ids_dict["del_node"].numel() != 0:
            pyg["del_node"].x = self.bert_model(
                torch.tensor(
                    token_ids_dict["del_node"].tolist(),
                    dtype=torch.long,
                    device=self.device,
                )
            )[0][:, 0, :]
        if token_ids_dict["add_node"].numel() == 0:
            pyg["add_node"].x = torch.zeros(
                (0, 768), dtype=torch.float, device=self.device
            )
        if token_ids_dict["del_node"].numel() == 0:
            pyg["del_node"].x = torch.zeros(
                (0, 768), dtype=torch.float, device=self.device
            )
        out0 = self.han_conv(pyg.x_dict, pyg.edge_index_dict)
        return torch.index_select(out0["del_node"], 0, delIndexes)

    def predict(self, pyg, delIndexes):
        token_ids_dict = pyg.token_ids_dict
        if token_ids_dict["add_node"].numel() != 0:
            pyg["add_node"].x = self.bert_model(
                torch.tensor(
                    token_ids_dict["add_node"].tolist(),
                    dtype=torch.long,
                    device=self.device,
                )
            )[0][:, 0, :]
        if token_ids_dict["del_node"].numel() != 0:
            pyg["del_node"].x = self.bert_model(
                torch.tensor(
                    token_ids_dict["del_node"].tolist(),
                    dtype=torch.long,
                    device=self.device,
                )
            )[0][:, 0, :]
        if token_ids_dict["add_node"].numel() == 0:
            pyg["add_node"].x = torch.zeros(
                (0, 768), dtype=torch.float, device=self.device
            )
        if token_ids_dict["del_node"].numel() == 0:
            pyg["del_node"].x = torch.zeros(
                (0, 768), dtype=torch.float, device=self.device
            )
        out0 = self.han_conv(pyg.x_dict, pyg.edge_index_dict)
        return torch.index_select(out0["del_node"], 0, delIndexes)


class rankNet(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.Linear(32, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 1),
        )

        self.output = nn.Sigmoid()

    def forward(self, input1, input2):
        s1 = self.model(input1)
        s2 = self.model(input2)
        return self.output(s1 - s2)

    def predict(self, input):
        return self.model(input)
