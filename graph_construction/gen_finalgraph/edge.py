"""
Represents an edge in the final heterogeneous graph.
"""
from typing import TYPE_CHECKING

# Forward reference to avoid circular imports
if TYPE_CHECKING:
    from .node import CGraphNode


class CGraphEdge:
    def __init__(self, src: 'CGraphNode', dst: 'CGraphNode', edge_type: str):
        self.src = src
        self.dst = dst
        self.type = edge_type

    def __str__(self) -> str:
        return f"({self.src.unit.getCodeStr()},{self.dst.unit.getCodeStr()},{self.type})"
