
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

from app.unit import CUnit, Position
from parse_joerndot.joern_node import JoernNode
from parse_joerndot.joern_edge import JoernEdge
from .node import CGraphNode
from .edge import CGraphEdge

class CGraphGenerator:
    """
    Generates heterogeneous graphs from C units and Joern CPG data.
    """
    
    def __init__(self, joern_nodes: List[JoernNode], joern_edges: List[JoernEdge], 
                 units: List[CUnit], file_name: str):
        self.joern_nodes = joern_nodes or []
        self.joern_edges = joern_edges or []
        self.units = units or []
        self.file_name = file_name
        
        # Graph components
        self.nodes = []
        self.edges = []
        self.special_nodes = [] 
        self.node_map = {}
        
        # Build the graph
        self._create_nodes()
        self._map_joern_to_nodes()
        self._build_graph()
        self._add_control_edges()
    
    def _create_nodes(self):
        """Create graph nodes from C units"""
        for i, unit in enumerate(self.units):
            node = CGraphNode(unit, i, self.file_name)
            self.nodes.append(node)
            
            # Identify special nodes that affect control flow
            code = unit.get_code_str()
            unit_type = unit.get_type()
            
            control_keywords_in_code = [
                "if", "else", "while", "for", "do", "switch", 
                "goto", "break", "continue", "return"
            ]
            
            has_control_keyword = any(keyword in code for keyword in control_keywords_in_code)
            
            control_unit_types = [
                "if", "else", "while", "for", "do", "dowhile", 
                "switch", "case", "default", "goto", "break", "continue", "return"
            ]
            
            is_control_unit_type = unit_type in control_unit_types
            
            if has_control_keyword or is_control_unit_type:
                self.special_nodes.append(node)
            
            
    def _map_joern_to_nodes(self):
        line_to_nodes = {} 
        
        for graph_node in self.nodes:
            unit_start = graph_node.unit.get_start_pos().line
            unit_end = graph_node.unit.get_end_pos().line
            
            for line in range(unit_start, unit_end + 1):
                if line not in line_to_nodes:
                    line_to_nodes[line] = []
                line_to_nodes[line].append(graph_node)
        
        for joern_node in self.joern_nodes:
            line_num = joern_node.line_num
            if line_num in line_to_nodes:
                self.node_map[joern_node.node_id] = line_to_nodes[line_num][0]

    def _build_graph(self):
        """Build graph edges from Joern edges"""
        
        cfg_count = 0
        ddg_count = 0
        skipped_type = 0
        skipped_missing = 0
        
        for edge in self.joern_edges:
            if edge.edge_type not in ["CFG", "DDG"]:
                skipped_type += 1
                continue

            src_node = self.node_map.get(edge.src)
            dst_node = self.node_map.get(edge.dst)
            if not src_node or not dst_node or src_node == dst_node:
                skipped_missing += 1
                continue

            if edge.edge_type == "CFG":
                if dst_node.index not in src_node.cfg_edges:
                    src_node.cfg_edges.append(dst_node.index)
                    dst_node.cfg_parents.append(src_node.index)
                self.edges.append(CGraphEdge(src_node, dst_node, "CFG"))
                cfg_count += 1

            elif edge.edge_type == "DDG":
                if dst_node.index not in src_node.dfg_edges:
                    src_node.dfg_edges.append(dst_node.index)
                    dst_node.dfg_parents.append(src_node.index)
                self.edges.append(CGraphEdge(src_node, dst_node, "DDG"))
                ddg_count += 1
    
    
    def _add_control_edges(self):
        """
        Add special control edges for branching constructs.
        """
        for special_node in self.special_nodes:
            if special_node.index + 1 >= len(self.nodes):
                continue
            
            next_node = self.nodes[special_node.index + 1]
            
            # Skip if there's already a direct CFG edge
            if (special_node.index + 1) in special_node.cfg_edges:
                continue
            
            # Redirect edges through the special node
            for parent_index in next_node.cfg_parents.copy():
                parent_node = self.nodes[parent_index]
                
                if next_node.index in parent_node.cfg_edges:
                    parent_node.cfg_edges.remove(next_node.index)
                
                if special_node.index not in parent_node.cfg_edges:
                    parent_node.cfg_edges.append(special_node.index)
                
                if parent_index not in special_node.cfg_parents:
                    special_node.cfg_parents.append(parent_index)
            
            if next_node.index not in special_node.cfg_edges:
                special_node.cfg_edges.append(next_node.index)
            
            next_node.cfg_parents.clear()
            next_node.cfg_parents.append(special_node.index)
    
    
    def get_nodes(self) -> List[CGraphNode]:
        self.nodes.sort(key=lambda n: n.unit.get_start_pos().line)
        return self.nodes
    
    def get_edges(self) -> List[CGraphEdge]:
        """Get all edges in the graph"""
        return self.edges
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the generated graph"""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "cfg_edges": len([e for e in self.edges if e.edge_type == "CFG"]),
            "dfg_edges": len([e for e in self.edges if e.edge_type == "DDG"]),
            "special_nodes": len(self.special_nodes),
            "units_processed": len(self.units),
            "joern_nodes_mapped": len([n for n in self.joern_nodes if n.node_id in self.node_map])
        } 