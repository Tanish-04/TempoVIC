from typing import List, Set, Optional, Dict, Iterable
from collections import OrderedDict

from app.unit import CUnit, Position
from .node import CGraphNode


class TrimGraph:
    def __init__(
        self,
        patchLines: List,
        nodes: List[CGraphNode],
        begIndex: int,
        fName: str,
        file_lines: Optional[List[str]] = None,
    ):
        self.patchLines = patchLines
        self.nodes = sorted(nodes, key=lambda n: n.unit.get_start_pos().line)
        
        self.begIndex = begIndex
        self.fName = fName
        self.topNodes: List[CGraphNode] = []
        self.finalNodes: List[CGraphNode] = []
        self.idMap = OrderedDict() 
        self.file_lines = file_lines or []
        self.node_line_map: Dict[int, Set[int]] = {}

        self.filterNodes()
        
        self.index_to_node = {n.index: n for n in self.nodes}

        visited = set()
        for topNode in self.finalNodes:
            set1 = set()
            set2 = set()
            orig_node = self.index_to_node[topNode.index]
            self.dfs1(topNode, orig_node, set1, visited)
            self.dfs2(topNode, orig_node, set2, visited)
            visited.add(topNode.index)

        self.compactNode()

    def getNode(self, pl) -> Optional[CGraphNode]:
        candidates = []
        
        # Find all nodes that contain this line
        for node in self.nodes:
            start_line = node.unit.get_start_pos().line
            end_line = node.unit.get_end_pos().line
            if start_line <= pl.lineno <= end_line:
                candidates.append(node)
        
        if not candidates:
            return None
        
        return min(candidates, key=lambda n: (
            n.unit.get_end_pos().line - n.unit.get_start_pos().line,
            abs(n.unit.get_start_pos().line - pl.lineno)
        ))

    def filterNodes(self):
        pre = None
        for l in self.patchLines:
            n = self.getNode(l)
            if n is None:
                pre = None
                continue

            if getattr(l, "lineno", 0) > 0:
                self.node_line_map.setdefault(n.index, set()).add(l.lineno)

            if n != pre:
                pre = n
                t = CGraphNode(
                    unit=n.unit,
                    index=n.index,
                    file_name=self.fName,
                    mapping_index=n.mapping_index 
                )
                if hasattr(t.unit, "set_node"):
                    t.unit.set_node(t)
                self.finalNodes.append(t)
                self.idMap[n.index] = n
        
        seen_spans = set()
        unique_nodes = []
        for node in self.finalNodes:
            span = (node.unit.get_start_pos().line, node.unit.get_end_pos().line)
            if span not in seen_spans:
                seen_spans.add(span)
                unique_nodes.append(node)
        
        self.finalNodes = unique_nodes

        # Trim unit code to only include patched lines for each node
        for node in self.finalNodes:
            relevant_lines = sorted(self.node_line_map.get(node.index, set()))
            if not relevant_lines:
                continue
            trimmed_unit = self._clone_unit_with_lines(node.unit, relevant_lines)
            node.unit = trimmed_unit

    def _clone_unit_with_lines(self, original_unit: CUnit, lines: Iterable[int]) -> CUnit:
        """
        Create a shallow clone of the unit whose code and span are trimmed
        to the provided line numbers.
        """
        line_list = [ln for ln in lines if ln > 0]
        if not line_list:
            return original_unit

        first_line = min(line_list)
        last_line = max(line_list)

        # Extract source text for the selected lines
        snippet_lines = []
        for ln in line_list:
            if 0 < ln <= len(self.file_lines):
                snippet_lines.append(self.file_lines[ln - 1].rstrip("\n"))

        if not snippet_lines:
            return original_unit

        trimmed_code = "\n".join(snippet_lines).rstrip()

        def leading_column(line: str) -> int:
            stripped = line.lstrip()
            if not stripped:
                return 1
            return len(line) - len(stripped) + 1

        start_column = leading_column(snippet_lines[0])
        end_column = len(snippet_lines[-1]) + 1 if snippet_lines[-1] else 1

        new_unit = CUnit(
            Position(first_line, start_column),
            Position(last_line, end_column),
            original_unit.get_type(),
            original_unit.get_end_char(),
        )
        new_unit.set_code(trimmed_code)
        new_unit.set_not_skip(original_unit.get_not_skip())
        new_unit.set_skip_ahead(original_unit.get_skip_ahead())
        new_unit.set_skip_two_sides(original_unit.get_skip_two_sides())
        new_unit.set_has_annotation(original_unit.get_has_annotation())

        # Copy declared names if any
        for field in original_unit.get_declared_field_names():
            new_unit.add_declared_field(field)
        for var in original_unit.get_declared_var_names():
            new_unit.add_declared_var(var)

        new_unit.cursor = getattr(original_unit, "cursor", None)

        return new_unit

    def dfs1(self, topNode: CGraphNode, curNode: CGraphNode, nodeSet: Set[int], visited: Set[int]):
        if curNode.index in nodeSet or curNode.index in visited:
            return

        nodeSet.add(curNode.index)

        for dst in curNode.cfg_edges:
            if dst in self.idMap:
                if dst != topNode.index and dst not in topNode.cfg_edges and dst not in visited:
                    topNode.cfg_edges.append(dst)
            else:
                self.dfs1(topNode, self.index_to_node[dst], nodeSet, visited) 

    def dfs2(self, topNode: CGraphNode, curNode: CGraphNode, nodeSet: Set[int], visited: Set[int]):
        if curNode.index in nodeSet or curNode.index in visited:
            return

        nodeSet.add(curNode.index)

        for dst in curNode.dfg_edges:
            if dst in self.idMap:
                if dst != topNode.index and dst not in topNode.dfg_edges and dst not in visited:
                    topNode.dfg_edges.append(dst)
            else:
                self.dfs2(topNode, self.index_to_node[dst], nodeSet, visited)  

    def compactNode(self):
        indexToIndex = {}
        indexToPos = {}

        for i in range(len(self.finalNodes)):
            indexToIndex[self.finalNodes[i].index] = self.begIndex + i
            indexToPos[self.begIndex + i] = i

        for n in self.finalNodes:
            n.index = indexToIndex[n.index]

            # Update CFG edges 
            for i in range(len(n.cfg_edges)):
                index = indexToIndex[n.cfg_edges[i]]
                pos = indexToPos[index]
                n.cfg_edges[i] = index
                self.finalNodes[pos].cfg_parents.append(n.index)

            # Update DFG edges
            for i in range(len(n.dfg_edges)):
                index = indexToIndex[n.dfg_edges[i]]
                pos = indexToPos[index]
                n.dfg_edges[i] = index
                self.finalNodes[pos].dfg_parents.append(n.index)

    def getFinalGraph(self) -> List[CGraphNode]:
        return self.finalNodes
