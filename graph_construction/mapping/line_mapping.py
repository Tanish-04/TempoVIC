from collections import deque
from typing import Dict
from .get_pos import GetPos
from .python_parser_generator import CParserGenerator
from .gumtree_bridge import create_line_mappings_for_c_files
from project.source_file import CSourceFile 
from config_loader import get_fake_libc_include

class LineMapping:
    def __init__(self, gumtree_path: str):
        self.type_set = {
            "function", "decl", "decl_stmt", "parameter", "parameter_list",
            "struct", "union", "enum", "typedef",
            "if", "if_stmt", "then", "else", "elseif",
            "for", "while", "do",
            "switch", "case", "default",
            "return", "break", "continue", "goto",
            "block", "block_content", "expr_stmt",
            "expr", "call", "operator",
            "condition",
            "type", "name", "literal",
            "argument_list", "argument",
            "include", "define", "directive",
        }
        self.gumtree_path = gumtree_path

        # AST contexts & helpers are prepared when files are known.
        self.tc1 = None
        self.tc2 = None
        self.gp1 = None
        self.gp2 = None
        self.trees1 = []
        self.trees2 = []

        # line -> line mappings (before -> after)
        self.mappings: Dict[int, int] = {}

    def _find_covering_node(self, graph_nodes, line: int):
        """
        Find the graph node (CGraphNode) whose unit spans a given line.
        """
        for n in graph_nodes:
            unit = n.unit
            start = unit.get_start_pos().line
            end = unit.get_end_pos().line
            if start <= line <= end:
                return n
        return None

    def generate_mappings(self, srcf1: CSourceFile, srcf2: CSourceFile):
        try:
            # Build AST contexts for GumTree-based mapping
            cgen = CParserGenerator()
            
            # After
            fake_libc = f"-I{get_fake_libc_include()}"
            self.tc1 = cgen.generate_from_string(srcf1.get_content(), clang_args=[fake_libc])
            self.tc2 = cgen.generate_from_string(srcf2.get_content(), clang_args=[fake_libc])
            self.gp1 = GetPos(srcf1.get_content())
            self.gp2 = GetPos(srcf2.get_content())

            # Collect filtered trees for both versions (sorted by line)
            self._process_trees()

            # Run GumTree matching and apply indices to Node objects
            self._gen_mappings(srcf1, srcf2)

            return self.mappings

        except Exception as e:
            import traceback
            print(f"\n[ERROR-LINEMAPPING] Exception in generate_mappings:")
            print(traceback.format_exc())
            raise RuntimeError(f"[LineMapping] Error while generating mappings: {e}")

    def _process_trees(self):
        """
        Collect nodes of interest from tc1/tc2 using your type_set, sorted by their line numbers.
        Only runs if tc1/tc2 & gp1/gp2 are available.
        """
        if not (self.tc1 and self.tc2 and self.gp1 and self.gp2):
            # Contexts not ready; nothing to do.
            self.trees1, self.trees2 = [], []
            return

        def collect(root, gp):
            q, result = deque([root]), []
            while q:
                t = q.popleft()
                result.append(t)
                try:
                    children = t.getChildren()
                except Exception:
                    children = None
                if not children:
                    continue
                q.extend(children)
            filtered = []
            for node in result:
                try:
                    pos = node.getPos()
                    if pos is None:
                        continue
                    gp.get_line_num(pos)
                except Exception:
                    continue
                filtered.append(node)
            return sorted(filtered, key=lambda x: gp.get_line_num(x.getPos()))

        self.trees1 = collect(self.tc1.get_root(), self.gp1)
        self.trees2 = collect(self.tc2.get_root(), self.gp2)

    @staticmethod
    def _node_start_line(n):
        """
        Get the starting line for a CGraphNode using Python unit APIs.
        Falls back gracefully if a variant naming is used.
        """
        u = getattr(n, "unit", None)
        if u is None:
            return None
        # Preferred: get_start_pos().line
        if hasattr(u, "get_start_pos"):
            sp = u.get_start_pos()
            if sp is not None and hasattr(sp, "line"):
                return sp.line
        # Java-style fallback: getBegPos().line
        if hasattr(u, "getBegPos"):
            bp = u.getBegPos()
            if bp is not None and hasattr(bp, "line"):
                return bp.line
        # Common attribute-style fallbacks
        if hasattr(u, "begin_line"):
            return u.begin_line
        if hasattr(n, "line"):
            return n.line
        return None

    @staticmethod
    def _node_type(n):
        """
        Get a comparable type string for a CGraphNode's unit.
        """
        u = getattr(n, "unit", None)
        if u is None:
            return None
        if hasattr(u, "get_type") and callable(u.get_type):
            return u.get_type()
        if hasattr(u, "type"):
            return u.type
        # Java-style fallback
        if hasattr(u, "getType") and callable(u.getType):
            try:
                t = u.getType()
                # Could be object with name
                if hasattr(t, "name"):
                    return t.name
                return str(t)
            except Exception:
                return None
        return None
    
    def _get_tree(self, node, cur_trees, gp):
        """
        Find the GumTree Tree that covers the given Node's position.
        Replicates Java's getTree() method (lines 203-235).
        
        Uses binary search to efficiently find the tree whose position range
        covers the node's beginning line.
        """
        beg_line = self._node_start_line(node)
        if beg_line is None or not cur_trees:
            return None
        
        # Binary search using Upperbound
        pos = self._upperbound_tree(cur_trees, beg_line, gp)
        
        # DEBUG: Show trees around the target line
        # Search backwards to find a tree that covers this node
        i = 1
        while (pos - i >= 0 and 
               gp.get_line_num(cur_trees[pos - i].getEndPos()) >= beg_line and
               gp.get_line_num(cur_trees[pos - i].getPos()) <= beg_line):
            i += 1
        i -= 1
        
        # Check if we found a covering tree
        if (pos - i >= 0 and pos - i < len(cur_trees) and
            gp.get_line_num(cur_trees[pos - i].getEndPos()) >= beg_line and
            gp.get_line_num(cur_trees[pos - i].getPos()) <= beg_line):
            return cur_trees[pos - i]
        
        # Check the tree at pos
        if pos >= len(cur_trees):
            return None
        
        end_line = node.unit.get_end_pos().line if hasattr(node, 'unit') else beg_line
        if end_line >= gp.get_line_num(cur_trees[pos].getPos()):
            return cur_trees[pos]
        
        return None
    
    def _get_node(self, beg_line, end_line, nodes, gp):
        """
        Find the Node that covers the given line range.
        Replicates Java's getNode() method (lines 237-259).
        
        Uses binary search to efficiently find the node whose position range
        covers the given line range.
        """
        if not nodes:
            return None
        
        # Binary search using Upperbound
        pos = self._upperbound_node(nodes, beg_line, gp)
        
        # Search backwards to find a node that covers this range
        i = 1
        while pos - i >= 0 and nodes[pos - i].unit.get_end_pos().line >= beg_line:
            i += 1
        i -= 1
        
        # Check if we found a covering node
        if (pos - i >= 0 and pos - i < len(nodes) and 
            nodes[pos - i].unit.get_end_pos().line >= beg_line):
            return nodes[pos - i]
        
        # Check the node at pos
        if pos >= len(nodes):
            return None
        
        if end_line >= nodes[pos].unit.get_start_pos().line:
            return nodes[pos]
        
        return None
    
    def _upperbound_tree(self, trees, val, gp):
        """
        Binary search to find upper bound in trees array.
        Replicates Java's Upperbound() method (lines 165-181).
        """
        l, u = 0, len(trees) - 1
        while l < u:
            midp = l + (u - l) // 2
            mid = gp.get_line_num(trees[midp].getPos())
            if val < mid:
                u = midp
            else:
                l = midp + 1
        
        if l >= len(trees):
            return len(trees)
        
        return l if val < gp.get_line_num(trees[l].getPos()) else len(trees)
    
    def _upperbound_node(self, nodes, val, gp):
        """
        Binary search to find upper bound in nodes array.
        Replicates Java's Upperbound1() method (lines 184-201).
        """
        l, u = 0, len(nodes) - 1
        while l < u:
            midp = l + (u - l) // 2
            mid = nodes[midp].unit.get_start_pos().line
            if val < mid:
                u = midp
            else:
                l = midp + 1
        
        if l >= len(nodes):
            return len(nodes)
        
        return l if val < nodes[l].unit.get_start_pos().line else len(nodes)
    
    def _gen_mappings(self, srcf1: CSourceFile, srcf2: CSourceFile):
        """
        Generate mappings using GumTree MappingStore.
        Replicates Java's genMappings() method (lines 261-309).
        
        Flow: Node → Tree → MappingStore → Tree → Node
        """
        try:
            before_path = srcf1.file_path
            after_path = srcf2.file_path

            # Get GumTree mappings (Tree-to-Tree at AST level)
            try:
                self.mappings = create_line_mappings_for_c_files(before_path, after_path)
            except Exception as gumtree_error:
                print(f"[WARN] GumTree mapping failed for {before_path} vs {after_path}: {gumtree_error}")
                self.mappings = {}

            # Get final nodes from both versions
            before_nodes = srcf1.get_final_nodes()
            after_nodes = srcf2.get_final_nodes()
            
            
            mapped_count = 0
            unmapped_count = 0

            # For each source node, find its mapping via GumTree
            # Replicates Java lines 269-298
            for idx, src_node in enumerate(before_nodes):
                src_start_line = self._node_start_line(src_node)
                
                # Find the GumTree Tree that corresponds to this Node
                t1 = self._get_tree(src_node, self.trees1, self.gp1)
                
                if t1 is None:
                    # FALLBACK: No tree found, try direct line mapping
                    if src_start_line not in self.mappings:
                        unmapped_count += 1
                        continue
                    
                    dst_line = self.mappings[src_start_line]
                    dst_node = self._get_node(dst_line, dst_line, after_nodes, self.gp2)
                    
                    if not dst_node:
                        unmapped_count += 1
                        continue
                    
                    src_type = self._node_type(src_node)
                    dst_type = self._node_type(dst_node)
                    
                    if src_type and dst_type and src_type == dst_type:
                        setattr(src_node, "mapping_index", dst_node.index)
                        setattr(dst_node, "mapping_index", src_node.index)
                        mapped_count += 1
                        continue
                    else:
                        unmapped_count += 1
                        continue
                
                # Check if GumTree matched this tree
                src_line = self.gp1.get_line_num(t1.getPos())
                if src_line not in self.mappings:
                    unmapped_count += 1
                    continue
                
                # Get the destination line from GumTree mapping
                dst_line = self.mappings[src_line]
                
                # Find the corresponding Node in the after version
                # Use tree's end position for range
                dst_end_line = self.gp2.get_line_num(self.mappings.get(self.gp1.get_line_num(t1.getEndPos()), dst_line))
                dst_node = self._get_node(dst_line, dst_end_line, after_nodes, self.gp2)
                
                # Verify type match (same as Java code line 277-278)
                if dst_node is not None:
                    src_type = self._node_type(src_node)
                    dst_type = self._node_type(dst_node)
                    
                    if src_type and dst_type and src_type == dst_type:
                        # Set bidirectional mapping
                        setattr(src_node, "mapping_index", dst_node.index)
                        setattr(dst_node, "mapping_index", src_node.index)
                        mapped_count += 1
                    else:
                        unmapped_count += 1
                else:
                    unmapped_count += 1

        except Exception as e:
            raise RuntimeError(f"[LineMapping] Error while generating mappings: {e}")



    # Debug helpers preserved
    def print_src_trees(self):
        for t in self.trees1:
            # print(t)
            pass

    def print_dst_trees(self):
        for t in self.trees2:
            # print(t)
            pass        

    def print_mappings(self):
        for b, a in sorted(self.mappings.items()):
            # print(f"{b} -> {a}")
            pass

    
