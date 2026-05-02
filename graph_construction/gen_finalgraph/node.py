from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

from app.unit import CUnit, Position
from parse_joerndot.joern_node import JoernNode
from parse_joerndot.joern_edge import JoernEdge

@dataclass
class CGraphNode:
    
    unit: CUnit
    index: int
    file_name: str
    is_del: bool = False
    
    # Control and data flow edges
    cfg_edges: List[int] = field(default_factory=list)
    dfg_edges: List[int] = field(default_factory=list)
    cfg_parents: List[int] = field(default_factory=list)
    dfg_parents: List[int] = field(default_factory=list)
    
    # Reference edges (field/variable and function references)
    field_edges: List[int] = field(default_factory=list)
    field_parents: List[int] = field(default_factory=list)
    method_edges: List[int] = field(default_factory=list)
    method_parents: List[int] = field(default_factory=list)
    
    # Mapping index for line mapping between before/after versions
    mapping_index: int = -1
    
    def clone(self) -> 'CGraphNode':
        """Create a deep copy of this node"""
        new_node = CGraphNode(
            unit=self.unit,
            index=self.index,
            file_name=self.file_name,
            is_del=False,
            mapping_index=self.mapping_index
        )
        
        # Copy edge lists
        new_node.cfg_edges = self.cfg_edges.copy()
        new_node.dfg_edges = self.dfg_edges.copy()
        new_node.cfg_parents = self.cfg_parents.copy()
        new_node.dfg_parents = self.dfg_parents.copy()
        new_node.field_edges = self.field_edges.copy()
        new_node.field_parents = self.field_parents.copy()
        new_node.method_edges = self.method_edges.copy()
        new_node.method_parents = self.method_parents.copy()
        
        return new_node
    
    def get_mapping_index(self) -> int:
        return self.mapping_index
    
    def set_mapping_index(self, i: int):
        self.mapping_index = i
    def __str__(self) -> str:
        return f"CGraphNode(index={self.index}, code='{self.unit.get_code_str()[:30]}...', is_del={self.is_del})"
