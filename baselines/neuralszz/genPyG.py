"""
Convert mini graph JSON format to PyTorch Geometric HeteroData format.

Creates a heterogeneous graph with:
  - Node types: del_node, add_node
  - Edge types: cdfg (CFG+DFG), ref (field/method parents), line_mapping
"""
import torch
from torch_geometric.data import HeteroData


def get_graph_data(graph):
    """Convert a mini graph (list of node dicts) to a PyG HeteroData object."""
    data = HeteroData()

    data["add_node"].x = torch.zeros((0, 64), dtype=torch.long)
    data["del_node"].x = torch.zeros((0, 64), dtype=torch.long)

    data["add_node"].token_ids = torch.zeros((0, 64), dtype=torch.long)
    data["del_node"].token_ids = torch.zeros((0, 64), dtype=torch.long)

    data["del_node", "line_mapping", "add_node"].edge_index = torch.zeros(
        (0, 2), dtype=torch.long
    )
    data["del_node", "cdfg", "del_node"].edge_index = torch.zeros(
        (0, 2), dtype=torch.long
    )
    data["del_node", "ref", "del_node"].edge_index = torch.zeros(
        (0, 2), dtype=torch.long
    )

    data["add_node", "line_mapping", "del_node"].edge_index = torch.zeros(
        (0, 2), dtype=torch.long
    )
    data["add_node", "cdfg", "add_node"].edge_index = torch.zeros(
        (0, 2), dtype=torch.long
    )
    data["add_node", "ref", "add_node"].edge_index = torch.zeros(
        (0, 2), dtype=torch.long
    )

    for node in graph:
        if node["isDel"]:
            data["del_node"].x = torch.cat(
                (data["del_node"].x, torch.tensor([node["token_ids"][:64]])), 0
            )
            data["del_node"].token_ids = torch.cat(
                (data["del_node"].token_ids, torch.tensor([node["token_ids"][:64]])), 0
            )
            for n in node["cfgs"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["del_node", "cdfg", "del_node"].edge_index = torch.cat(
                    (data["del_node", "cdfg", "del_node"].edge_index, edge), 0
                )

            for n in node["dfgs"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["del_node", "cdfg", "del_node"].edge_index = torch.cat(
                    (data["del_node", "cdfg", "del_node"].edge_index, edge), 0
                )

            for n in node["fieldParents"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["del_node", "ref", "del_node"].edge_index = torch.cat(
                    (data["del_node", "ref", "del_node"].edge_index, edge), 0
                )

            for n in node["methodParents"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["del_node", "ref", "del_node"].edge_index = torch.cat(
                    (data["del_node", "ref", "del_node"].edge_index, edge), 0
                )

            if node["lineMapIndex"] != -1:
                edge = torch.tensor(
                    [node["nodeIndex"], node["lineMapIndex"]], dtype=torch.long
                ).view(1, -1)
                data["del_node", "line_mapping", "add_node"].edge_index = torch.cat(
                    (data["del_node", "line_mapping", "add_node"].edge_index, edge), 0
                )

        else:
            data["add_node"].x = torch.cat(
                (data["add_node"].x, torch.tensor([node["token_ids"][:64]])), 0
            )
            data["add_node"].token_ids = torch.cat(
                (data["add_node"].token_ids, torch.tensor([node["token_ids"][:64]])), 0
            )
            for n in node["cfgs"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data[
                    "add_node", "cdfg", "add_node"
                ].dtype = torch.longedge_index = torch.cat(
                    (data["add_node", "cdfg", "add_node"].edge_index, edge), 0
                )

            for n in node["dfgs"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["add_node", "cdfg", "add_node"].edge_index = torch.cat(
                    (data["add_node", "cdfg", "add_node"].edge_index, edge), 0
                )

            for n in node["fieldParents"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["add_node", "ref", "add_node"].edge_index = torch.cat(
                    (data["add_node", "ref", "add_node"].edge_index, edge), 0
                )

            for n in node["methodParents"]:
                edge = torch.tensor([node["nodeIndex"], n], dtype=torch.long).view(
                    1, -1
                )
                data["add_node", "ref", "add_node"].edge_index = torch.cat(
                    (data["add_node", "ref", "add_node"].edge_index, edge), 0
                )

            if node["lineMapIndex"] != -1:
                edge = torch.tensor(
                    [node["nodeIndex"], node["lineMapIndex"]], dtype=torch.long
                ).view(1, -1)
                data["add_node", "line_mapping", "del_node"].edge_index = torch.cat(
                    (data["add_node", "line_mapping", "del_node"].edge_index, edge), 0
                )

    data["del_node", "line_mapping", "add_node"].edge_index = (
        data["del_node", "line_mapping", "add_node"].edge_index.t().contiguous()
    )
    data["del_node", "cdfg", "del_node"].edge_index = (
        data["del_node", "cdfg", "del_node"].edge_index.t().contiguous()
    )
    data["del_node", "ref", "del_node"].edge_index = (
        data["del_node", "ref", "del_node"].edge_index.t().contiguous()
    )

    data["add_node", "line_mapping", "del_node"].edge_index = (
        data["add_node", "line_mapping", "del_node"].edge_index.t().contiguous()
    )
    data["add_node", "cdfg", "add_node"].edge_index = (
        data["add_node", "cdfg", "add_node"].edge_index.t().contiguous()
    )
    data["add_node", "ref", "add_node"].edge_index = (
        data["add_node", "ref", "add_node"].edge_index.t().contiguous()
    )
    return data
