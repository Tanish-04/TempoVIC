import os
from typing import List, Set

from app.clang_parser import ClangParser
from app.unit import CUnit, Position
from parse_joerndot.joern_graph import JoernGraph
from gen_finalgraph.gen_graph import CGraphGenerator
from gen_finalgraph.node import CGraphNode
from gen_finalgraph.trim_graph import TrimGraph
from parse_patch.patch_line import PatchLine
from .clang_function_tracker import ClangFunctionTracker
from clang.cindex import CursorKind
from .method import Method
from .call import Call

class CSourceFile:
    """
    Processes a single C source file and its corresponding Joern data.
    """

    def __init__(self, file_path: str, joern_path: str, node_index_offset: int,
                 patch_lines: List[PatchLine], file_name: str):
        self.file_path = file_path
        self.joern_path = joern_path
        self.file_name = file_name
        self.node_index_offset = node_index_offset
        self.patch_lines = patch_lines

        self.units = []
        self.file_lines = []
        self.joern_nodes = []
        self.joern_edges = []
        self.final_nodes: List[CGraphNode] = []
        self.function_tracker: ClangFunctionTracker = None

        self.method_decls = []
        self.field_decls = []
        
        self.final_methods = []
        self.final_calls = []

        self._process_file()

    def _process_file(self):
        try:
            print(f"Parsing C file: {self.file_path}")
            c_parser = ClangParser(self.file_path)
            self.units = c_parser.get_all_units() or []
            self.file_lines = getattr(c_parser, "file_content", []) or []

            patch_line_numbers = {pl.lineno for pl in self.patch_lines if getattr(pl, "lineno", 0) > 0}
            
            interesting_units = []
            
            for unit in self.units:
                start_line = unit.get_start_pos().line
                end_line = unit.get_end_pos().line
                if any(start_line <= ln <= end_line for ln in patch_line_numbers):
                    interesting_units.append(unit)

            expr_units = [u for u in self.units if u.get_type() == "ExpressionStatement"]
            
            print(f"[UNIT-LIST] {self.file_name}: total units={len(self.units)}, "
                  f"expression statements={len(expr_units)}, units covering patch lines={len(interesting_units)}; "
                  f"patch lines={sorted(patch_line_numbers)}")
            
            for unit in expr_units:
                print(f"    [EXPR] {unit.get_start_pos().line}-{unit.get_end_pos().line}: {unit.get_code_str()}")
            for unit in interesting_units:
                print(f"    [UNIT] {unit.get_type()} {unit.get_start_pos().line}-{unit.get_end_pos().line}: {unit.get_code_str()}")

            print(f"[UNIT-LIST] {self.file_name}: total units={len(self.units)}, "
                  f"units covering patch lines={len(interesting_units)}; patch lines={sorted(patch_line_numbers)}")
            
            for unit in interesting_units:
                print(f"    [UNIT] {unit.get_type()} {unit.get_start_pos().line}-{unit.get_end_pos().line}: {unit.get_code_str()}")


            if c_parser.ast is not None:
                self.get_all_method_declarations(c_parser.ast)
                self.get_all_field_declarations(c_parser.ast)
            
            print(f"Parsing Joern data from: {self.joern_path}")
            self._parse_joern_data()

            graph_gen = CGraphGenerator(self.joern_nodes, self.joern_edges, self.units, self.file_name)
            nodes = graph_gen.get_nodes()

            # method/call extraction (must attach to nodes)
            if c_parser.ast is not None:
                self.function_tracker = ClangFunctionTracker(self.file_path, c_parser.ast)
                self.function_tracker.build_method_edges(nodes)  # attach by line spans
                self.final_methods = self.function_tracker.get_functions()
                self.final_calls = self.function_tracker.get_function_calls()

            # always trim graph
            gen_graph = CGraphGenerator(self.joern_nodes, self.joern_edges, self.units, self.file_name)
            nodes1 = gen_graph.get_nodes()
            trimmer = TrimGraph(
                self.patch_lines,
                nodes1,
                self.node_index_offset,
                self.file_name,
                file_lines=self.file_lines,
            )
            self.final_nodes = trimmer.getFinalGraph()
            self._ensure_patch_line_nodes()
            self._sanitize_node_codes()

            # field references 
            self._gen_field_ref()

        except Exception as e:
            import traceback
            print(f"Error processing file {self.file_path}: {e}")
            traceback.print_exc()
            self.final_nodes = []

    def _parse_joern_data(self):
        """Parse Joern DOT files from joern directory"""
        if not os.path.isdir(self.joern_path):
            return

        # Load ALL Joern files in the directory (they're named by function, not source file)
        for item in os.listdir(self.joern_path):
            file_path = os.path.join(self.joern_path, item)
            if not os.path.isfile(file_path):
                continue
            
            # Skip .c files (those are source copies)
            if item.endswith('.c'):
                continue
                
            try:
                joern_parser = JoernGraph(file_path)
                nodes = joern_parser.get_nodes()
                edges = joern_parser.get_edges()
                self.joern_nodes.extend(nodes)
                self.joern_edges.extend(edges)
                
            except Exception as e:
                # skip files that can't be parsed
                pass
        

    def _gen_field_ref(self):
        """Add field reference edges."""
        field_nodes = {}
        for n in self.final_nodes:
            if n.unit.is_field_declaration():
                for fname in n.unit.get_declared_field_names():
                    field_nodes[fname] = n.index

        for n in self.final_nodes:
            if not n.unit.is_statement():
                continue
            code = n.unit.get_code_str()
            for fname, fidx in field_nodes.items():
                if fname in code and n.index != fidx:
                    if not n.field_parents:
                        n.field_parents.append(fidx)
                    field_node = next(fn for fn in self.final_nodes if fn.index == fidx)
                    if n.index not in field_node.field_edges:
                        field_node.field_edges.append(n.index)

    def _ensure_patch_line_nodes(self):
        """Ensure every patch line with content has a corresponding graph node."""

        if not self.file_lines:
            return

        covered_lines: Set[int] = set()
        for node in self.final_nodes:
            start = node.unit.get_start_pos().line
            end = node.unit.get_end_pos().line
            for ln in range(start, end + 1):
                covered_lines.add(ln)

        patch_lines_by_number = {
            getattr(pl, "lineno", 0): pl for pl in self.patch_lines if getattr(pl, "lineno", 0) > 0
        }

        if not patch_lines_by_number:
            return

        next_index = max((node.index for node in self.final_nodes), default=self.node_index_offset - 1) + 1

        for line_no in sorted(patch_lines_by_number.keys()):
            if line_no in covered_lines:
                continue
            if line_no > len(self.file_lines):
                continue
            source_line = self.file_lines[line_no - 1].rstrip("\n")
            if not source_line.strip():
                continue
            
            # Skip comment-only lines
            if self._is_comment_only_line(source_line):
                continue

            unit_type = self._infer_unit_type(source_line)
            start_col = self._leading_column(source_line)
            end_col = len(source_line.rstrip()) + 1 if source_line.rstrip() else start_col

            unit = CUnit(
                Position(line_no, start_col),
                Position(line_no, end_col),
                unit_type,
            )
            unit.set_code(self._sanitize_code_text(source_line))

            if not unit.get_code_str():
                continue

            node = CGraphNode(unit=unit, index=next_index, file_name=self.file_name)
            self.final_nodes.append(node)
            covered_lines.add(line_no)
            next_index += 1

    def _sanitize_node_codes(self):
        """Remove stray braces from node code strings."""
        for node in self.final_nodes:
            code = node.unit.get_code_str()
            if not code:
                continue
            sanitized = self._sanitize_code_text(code)
            node.unit.set_code(sanitized)

    @staticmethod
    def _is_comment_only_line(line: str) -> bool:
        """
        Check if a line contains only comments (no executable code).
        
        Returns True if the line is:
        - A C-style comment line: /* ... */ or /* ... or ... */ or */
        - A C++ style comment: // ...
        - Part of a multi-line comment: * ... or */
        - Only whitespace
        """
        import re
        
        stripped = line.strip()
        
        if not stripped:
            return True
        
        # Check for common comment patterns
        comment_patterns = [
            r'^/\*.*\*/$',       # Single-line /* comment */
            r'^/\*',             # Start of multi-line comment /* ...
            r'^\*[^/]',          # Middle of multi-line comment * ...
            r'^\*/',             # End of multi-line comment */
            r'^//',              # Single-line C++ comment // ...
            r'^\*\s*$',          # Just a * with optional whitespace
        ]
        
        for pattern in comment_patterns:
            if re.match(pattern, stripped):
                return True
        
        return False

    @staticmethod
    def _leading_column(line: str) -> int:
        stripped = line.lstrip()
        if not stripped:
            return 1
        return len(line) - len(stripped) + 1

    @staticmethod
    def _infer_unit_type(code_line: str) -> str:
        stripped = code_line.lstrip()
        if stripped.startswith("if "):
            return "IfStatement"
        if stripped.startswith("while "):
            return "WhileStatement"
        if stripped.startswith("for "):
            return "ForStatement"
        if stripped.startswith("return"):
            return "ReturnStatement"
        return "ExpressionStatement"

    @staticmethod
    def _sanitize_code_text(code: str) -> str:
        lines = code.splitlines() if "\n" in code else [code]
        cleaned_lines = []
        for text in lines:
            stripped = text.strip()
            if not stripped:
                continue
            if stripped in ("{", "}"):
                continue
            if stripped.endswith("{"):
                idx = text.rfind("{")
                text = text[:idx].rstrip()
                stripped = text.strip()
                if not stripped:
                    continue
            if stripped.startswith("}"):
                idx = text.find("}")
                text = text[idx + 1 :].lstrip()
                stripped = text.strip()
                if not stripped:
                    continue
            cleaned_lines.append(stripped)
        return " ".join(cleaned_lines).strip()

    def get_all_method_declarations(self, node):
        """ getAllMethodDeclaration analog: collect & sort function definitions."""
        methods = []

        def visit(cur):
            # FUNCTION_DECL that is a *definition* (has a body)
            if cur.kind == CursorKind.FUNCTION_DECL and cur.is_definition():
                try:
                    sl = cur.extent.start.line
                    el = cur.extent.end.line
                except Exception:
                    sl = getattr(cur.location, "line", 0) or 0
                    el = sl
                name = cur.spelling or ""
                # Count parameters (some clang builds use get_arguments())
                try:
                    param_count = sum(1 for _ in cur.get_arguments())
                except Exception:
                    param_count = 0
                methods.append({
                    "name": name,
                    "begin": sl,
                    "end": el,
                    "param_count": param_count,
                    "cursor": cur
                })

            # Recurse
            for ch in cur.get_children():
                visit(ch)

            visit(ast.cursor)

            # Sort by begin line
            methods.sort(key=lambda m: m["begin"])
            self.method_decls = methods

    def get_all_field_declarations(self, ast):
        """
        For C, we treat 'fields' as struct/union members (FIELD_DECL).
        We also optionally include TU-scope globals (VAR_DECL with parent TU) if helpful.
        we group FIELD_DECLs by their parent struct/union.
        """
        groups = []   # each group ~ one struct/union, carrying all its field names
        globals_ = [] # optional: file-scope globals (VAR_DECL under TU)

        def collect_struct_fields(agg_cursor):
            names = []
            for ch in agg_cursor.get_children():
                if ch.kind == CursorKind.FIELD_DECL and ch.spelling:
                    names.append(ch.spelling)
            if names:
                try:
                    sl = agg_cursor.extent.start.line
                    el = agg_cursor.extent.end.line
                except Exception:
                    sl = getattr(agg_cursor.location, "line", 0) or 0
                    el = sl
                groups.append({
                    "begin": sl,
                    "end": el,
                    "names": names,         # list of field names
                    "cursor": agg_cursor,
                    "kind": agg_cursor.kind # STRUCT_DECL or UNION_DECL
                })

        def visit(cur):
            k = cur.kind
            if k in (CursorKind.STRUCT_DECL, CursorKind.UNION_DECL):
                collect_struct_fields(cur)

            # Optionally include file-scope globals to behave more like “fields”
            if k == CursorKind.VAR_DECL and cur.semantic_parent and cur.semantic_parent.kind == CursorKind.TRANSLATION_UNIT:
                try:
                    sl = cur.extent.start.line
                    el = cur.extent.end.line
                except Exception:
                    sl = getattr(cur.location, "line", 0) or 0
                    el = sl
                if cur.spelling:
                    globals_.append({
                        "begin": sl,
                        "end": el,
                        "names": [cur.spelling],  # single name
                        "cursor": cur,
                        "kind": k
                    })

            for ch in cur.get_children():
                visit(ch)

        visit(ast.cursor)

        # Merge groups + (optionally) globals and sort by begin line
        field_decls = groups + globals_
        field_decls.sort(key=lambda g: g["begin"])
        self.field_decls = field_decls

    def _get_node(self, beg_line: int, end_line: int):
        """Find the first node whose unit.begin line is in [beg_line, end_line]."""
        l, r = 0, len(self.final_nodes) - 1
        while l < r:
            mid = (l + r) // 2
            if self.final_nodes[mid].unit.get_start_pos().line >= beg_line:
                r = mid
            else:
                l = mid + 1

        if l == len(self.final_nodes):
            return None

        uline = self.final_nodes[l].unit.get_start_pos().line
        if beg_line <= uline <= end_line:
            return self.final_nodes[l]
        return None

   
    def get_methods(self):
        return self.final_methods

    def get_calls(self):
        return self.final_calls


    def get_content(self) -> str:
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {self.file_path}: {e}")
            return ""
    
    def get_file_name(self) -> str:
        return self.file_name
    
    def get_final_nodes(self) -> List[CGraphNode]:
        return self.final_nodes
