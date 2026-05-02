from typing import Dict, List, Tuple
import clang.cindex
from clang.cindex import CursorKind

class ClangExtract:
    _CACHE_ATTR = "_clang_extract_cache"

    @staticmethod
    def _ensure_cache(translation_unit) -> Dict:
        """
        Build (or reuse) a cache of nodes grouped by CursorKind for a translation unit.
        The cache is stored directly on the TranslationUnit object to avoid repeated
        full-tree traversals for each extractor method.
        """
        if translation_unit is None or translation_unit.cursor is None:
            return {"by_kind": {}, "expressions": [], "control": []}

        cache = getattr(translation_unit, ClangExtract._CACHE_ATTR, None)
        if cache is not None:
            return cache

        by_kind: Dict[CursorKind, List] = {}
        expressions: List = []
        control_structures: List = []
        expr_kinds = {
            CursorKind.CALL_EXPR,
            CursorKind.BINARY_OPERATOR,
            CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,
            CursorKind.UNARY_OPERATOR,
            CursorKind.CONDITIONAL_OPERATOR,
            CursorKind.CXX_THROW_EXPR,
            CursorKind.CXX_NEW_EXPR,
            CursorKind.CXX_DELETE_EXPR,
            CursorKind.CSTYLE_CAST_EXPR,
            CursorKind.INIT_LIST_EXPR,
            CursorKind.ARRAY_SUBSCRIPT_EXPR,
            CursorKind.MEMBER_REF_EXPR,
        }
        control_kinds = {
            CursorKind.IF_STMT,
            CursorKind.WHILE_STMT,
            CursorKind.FOR_STMT,
            CursorKind.DO_STMT,
            CursorKind.SWITCH_STMT,
        }

        stack = [translation_unit.cursor]
        while stack:
            cursor = stack.pop()
            if cursor is None or cursor.kind is None:
                continue

            by_kind.setdefault(cursor.kind, []).append(cursor)
            if cursor.kind in expr_kinds:
                expressions.append(cursor)
            if cursor.kind in control_kinds:
                control_structures.append(cursor)

            for ch in cursor.get_children():
                stack.append(ch)

        cache = {
            "by_kind": by_kind,
            "expressions": expressions,
            "control": control_structures,
        }
        setattr(translation_unit, ClangExtract._CACHE_ATTR, cache)
        return cache

    @staticmethod
    def _filter_by_window(cursors: List, line_window: Tuple[int, int, int] = None) -> List:
        """Restrict cursor list to those intersecting the provided line window."""
        if not cursors or not line_window:
            return list(cursors) if cursors else []

        start_line, end_line, padding = line_window
        low = max(1, start_line - padding)
        high = end_line + padding if end_line else float("inf")

        filtered = []
        for cursor in cursors:
            try:
                c_start = cursor.extent.start.line
                c_end = cursor.extent.end.line
            except Exception:
                filtered.append(cursor)
                continue

            if c_end < low or c_start > high:
                continue
            filtered.append(cursor)
        return filtered

    @staticmethod
    def _filter_main_file(translation_unit, cursors: List) -> List:
        if not cursors or translation_unit is None:
            return cursors or []

        main_file = translation_unit.spelling if translation_unit.spelling else None
        if not main_file:
            return cursors

        filtered = []
        for cursor in cursors:
            try:
                loc_file = cursor.location.file
                if loc_file is not None and loc_file.name != main_file:
                    continue
                if loc_file is None and cursor.extent and cursor.extent.start.file:
                    if cursor.extent.start.file.name != main_file:
                        continue
            except Exception:
                continue
            filtered.append(cursor)
        return filtered

    @staticmethod
    def _visit_tu(translation_unit, want_kinds: set, debug_range=None, line_window=None) -> List:
        """Collect cursors of the given kinds, restricted to the main file."""
        results = []
        nodes_visited = 0
        nodes_skipped_file = 0
        nodes_skipped_invalid = 0
        
        # Debug: track what we see in the target range
        if debug_range:
            debug_start, debug_end = debug_range
            nodes_in_range = []

        def visit(cursor):
            nonlocal nodes_visited, nodes_skipped_file, nodes_skipped_invalid
            try:
                nodes_visited += 1
                
                # Skip invalid / null cursors
                if cursor is None or cursor.kind is None:
                    nodes_skipped_invalid += 1
                    return
                
                # Debug: track nodes in target range
                if debug_range and cursor.location and cursor.location.line:
                    line = cursor.location.line
                    if debug_start <= line <= debug_end:
                        nodes_in_range.append((line, cursor.kind.name, cursor.spelling[:30] if cursor.spelling else ""))
                
                # Restrict to the main file when locations exist
                if cursor.location and cursor.location.file:
                    if str(cursor.location.file) != translation_unit.spelling:
                        nodes_skipped_file += 1
                        # still descend—macro groups can nest—but don't collect
                        for ch in cursor.get_children():
                            visit(ch)
                        return
                if cursor.kind in want_kinds:
                    results.append(cursor)
                for ch in cursor.get_children():
                    visit(ch)
            except Exception as e:
                # robust to any libclang oddities
                pass

        if debug_range is not None:
            if translation_unit and translation_unit.cursor:
                visit(translation_unit.cursor)

            if nodes_in_range:
                print(f"  Nodes in range {debug_start}-{debug_end}:")
                for line, kind, spelling in nodes_in_range[:10]:
                    print(f"    Line {line}: {kind} '{spelling}'")
            else:
                print(f"  NO NODES FOUND in range {debug_start}-{debug_end}")
            return ClangExtract._filter_by_window(results, line_window)

        cache = ClangExtract._ensure_cache(translation_unit)
        collected = []
        for kind in want_kinds:
            collected.extend(cache["by_kind"].get(kind, []))
        collected = ClangExtract._filter_main_file(translation_unit, collected)
        return ClangExtract._filter_by_window(collected, line_window)

    @staticmethod
    def get_function_definitions(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.FUNCTION_DECL}, line_window=line_window
        )

    @staticmethod
    def get_declarations(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.VAR_DECL}, line_window=line_window
        )

    @staticmethod
    def get_field_declarations(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.FIELD_DECL}, line_window=line_window
        )

    @staticmethod
    def get_struct_declarations(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.STRUCT_DECL}, line_window=line_window
        )

    @staticmethod
    def get_union_declarations(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.UNION_DECL}, line_window=line_window
        )

    @staticmethod
    def get_enum_declarations(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.ENUM_DECL}, line_window=line_window
        )

    @staticmethod
    def get_enum_constants(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.ENUM_CONSTANT_DECL}, line_window=line_window
        )

    @staticmethod
    def get_typedef_declarations(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.TYPEDEF_DECL}, line_window=line_window
        )

    @staticmethod
    def get_if_statements(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.IF_STMT}, line_window=line_window
        )
    
    @staticmethod
    def get_else_statements(translation_unit, line_window=None) -> List:
        """ Extract ELSE clauses from IF_STMT nodes. """
        if_stmts = ClangExtract._visit_tu(translation_unit, {CursorKind.IF_STMT}, line_window=line_window)
        else_clauses = []
        
        for if_stmt in if_stmts:
            children = list(if_stmt.get_children())
            
            if len(children) >= 3:
                # Children are typically: [condition, then-body, else-body]
                # But sometimes there can be extra nodes (implicit declarations)
                # The safest approach: else-body is the LAST child after then-body
                else_body = children[-1]  # Last child
                
                # Verify it's actually a statement (not a declaration or expression)
                if else_body.kind in (CursorKind.COMPOUND_STMT, CursorKind.RETURN_STMT,
                                     CursorKind.BREAK_STMT, CursorKind.CONTINUE_STMT,
                                     CursorKind.DECL_STMT, CursorKind.IF_STMT,
                                     CursorKind.FOR_STMT, CursorKind.WHILE_STMT,
                                     CursorKind.DO_STMT, CursorKind.SWITCH_STMT):
                    else_clauses.append(else_body)
                            
        return else_clauses

    @staticmethod
    def get_while_statements(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.WHILE_STMT}, line_window=line_window
        )

    @staticmethod
    def get_for_statements(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.FOR_STMT}, line_window=line_window
        )

    @staticmethod
    def get_do_while_statements(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.DO_STMT}, line_window=line_window
        )

    @staticmethod
    def get_switch_statements(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.SWITCH_STMT}, line_window=line_window
        )

    @staticmethod
    def get_switch_entries(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.CASE_STMT, CursorKind.DEFAULT_STMT}, line_window=line_window
        )
    
    @staticmethod
    def get_return_statements(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.RETURN_STMT}, line_window=line_window
        )

    @staticmethod
    def get_break_statements(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.BREAK_STMT}, line_window=line_window
        )

    @staticmethod
    def get_continue_statements(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.CONTINUE_STMT}, line_window=line_window
        )
    
    @staticmethod
    def get_decl_statements(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.DECL_STMT}, line_window=line_window
        )

    @staticmethod
    def get_goto_statements(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.GOTO_STMT}, line_window=line_window
        )

    @staticmethod
    def get_label_statements(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.LABEL_STMT}, line_window=line_window
        )

    @staticmethod
    def get_compound_statements(translation_unit, line_window=None) -> List:
        return ClangExtract._visit_tu(
            translation_unit, {CursorKind.COMPOUND_STMT}, line_window=line_window
        )

    @staticmethod
    def get_includes(translation_unit) -> List[Tuple[str, int, int, int, int, str]]:
        results = []
        path = translation_unit.spelling if translation_unit else None
        if not path:
            return results
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, raw in enumerate(f, 1):
                    s = raw.lstrip()
                    if s.startswith("#include"):
                        col = raw.find('#') + 1 if '#' in raw else 1
                        results.append(("IncludeDirective", i, col, i, len(raw) + 1, raw.rstrip('\n')))
        except Exception:
            pass
        return results
    
    @staticmethod
    def get_expression_statements(translation_unit, line_window=None) -> List:
        from clang.cindex import CursorKind
        # a union of common expression cursor kinds
        want = {
            CursorKind.CALL_EXPR,
            CursorKind.BINARY_OPERATOR,
            CursorKind.COMPOUND_ASSIGNMENT_OPERATOR, 
            CursorKind.UNARY_OPERATOR,
            CursorKind.CONDITIONAL_OPERATOR,
        }

        results = []
        
        # Build a set to track which expressions contain others (for filtering nested ones)
        cache = ClangExtract._ensure_cache(translation_unit)
        all_expressions_raw = cache["expressions"]
        print(f"[EXPR-DEBUG] Raw expressions count: {len(all_expressions_raw)}")
        all_expressions = [
            expr for expr in ClangExtract._filter_main_file(translation_unit, all_expressions_raw)
            if expr.kind in want
        ]
        print(f"[EXPR-DEBUG] Filtered expressions count: {len(all_expressions)}")
        all_control_structures = ClangExtract._filter_main_file(translation_unit, cache["control"])

        control_header_nodes = []
        for ctrl in all_control_structures:
            try:
                children = list(ctrl.get_children())
            except Exception:
                continue

            if not children:
                continue

            if ctrl.kind == CursorKind.IF_STMT:
                header_candidates = children[:1]
            elif ctrl.kind == CursorKind.WHILE_STMT:
                header_candidates = children[:1]
            elif ctrl.kind == CursorKind.FOR_STMT:
                header_candidates = children[:3]
            elif ctrl.kind == CursorKind.DO_STMT:
                header_candidates = children[-1:]
            elif ctrl.kind == CursorKind.SWITCH_STMT:
                header_candidates = children[:1]
            else:
                header_candidates = []

            for candidate in header_candidates:
                if candidate is not None:
                    control_header_nodes.append(candidate)

        def cursor_contains(outer, inner):
            try:
                outer_start = outer.extent.start
                outer_end = outer.extent.end
                inner_start = inner.extent.start
                inner_end = inner.extent.end
            except Exception:
                return False

            starts_before_or_same = (
                (outer_start.line < inner_start.line)
                or (outer_start.line == inner_start.line and outer_start.column <= inner_start.column)
            )
            ends_after_or_same = (
                (outer_end.line > inner_end.line)
                or (outer_end.line == inner_end.line and outer_end.column >= inner_end.column)
            )
            return starts_before_or_same and ends_after_or_same

        def is_control_header_expression(expr):
            for header in control_header_nodes:
                if header == expr or cursor_contains(header, expr):
                    return True
            return False
        
        def is_nested_in_another_expression(c):
            """
            Check if this expression is nested inside another expression OR
            is a condition/initializer of a control structure.
            """
            try:
                my_start = c.extent.start.line
                my_end = c.extent.end.line
                my_start_col = c.extent.start.column
                my_end_col = c.extent.end.column
            except:
                return False
            
            # Check if nested in another expression
            for other in all_expressions:
                if other == c:
                    continue
                
                try:
                    other_start = other.extent.start.line
                    other_end = other.extent.end.line
                    other_start_col = other.extent.start.column
                    other_end_col = other.extent.end.column
                except:
                    continue
                
                # Check if 'other' contains 'c'
                starts_before_or_same = (other_start < my_start) or (other_start == my_start and other_start_col <= my_start_col)
                ends_after_or_same = (other_end > my_end) or (other_end == my_end and other_end_col >= my_end_col)
                
                if starts_before_or_same and ends_after_or_same:
                    return True 
            
            if is_control_header_expression(c):
                return True
            
            return False
        
        # Second pass: filter out nested expressions
        for expr in ClangExtract._filter_by_window(all_expressions, line_window):
            if not is_nested_in_another_expression(expr):
                results.append(expr)

        for expr in results:
            try:
                start = expr.extent.start
                end = expr.extent.end
            except Exception:
                print(f"[EXPR-DEBUG] kind={expr.kind} span <unknown> spelling={expr.spelling}")

        return results

    # Preprocessor / Macros (for “macro switches”) 
    @staticmethod
    def get_macro_conditionals(translation_unit, line_window=None) -> List:
        """
        Collect preprocessor conditionals so we can model macro-driven control flow:
         - #if, #ifdef, #ifndef, #elif, #else, #endif
        """
        want = set()
        # These kinds exist in modern libclang (LLVM 10+); LLVM 18 supports them.
        for name in (
            "IF_DIRECTIVE",
            "IFDEF_DIRECTIVE",
            "IFNDEF_DIRECTIVE",
            "ELIF_DIRECTIVE",
            "ELSE_DIRECTIVE",
            "ENDIF_DIRECTIVE",
        ):
            kind = getattr(CursorKind, name, None)
            if kind is not None:
                want.add(kind)
        return ClangExtract._visit_tu(translation_unit, want, line_window=line_window)

    @staticmethod
    def get_macro_instantiations(translation_unit, line_window=None) -> List:
        """Extract macro instantiations from Clang AST"""
        macros = []
        try:
            def visit_cursor(cursor):
                if cursor.kind == CursorKind.MACRO_INSTANTIATION:
                    if cursor.location.file and str(cursor.location.file) == translation_unit.spelling:
                        if line_window:
                            start_line = cursor.extent.start.line if cursor.extent else cursor.location.line
                            end_line = cursor.extent.end.line if cursor.extent else cursor.location.line
                            low = max(1, line_window[0] - line_window[2])
                            high = line_window[1] + line_window[2] if line_window[1] else float("inf")
                            if end_line < low or start_line > high:
                                return
                        macros.append(cursor)
                for child in cursor.get_children():
                    visit_cursor(child)

            if translation_unit and translation_unit.cursor:
                visit_cursor(translation_unit.cursor)
        except Exception:
            pass
        return macros

    @staticmethod
    def get_macro_definitions(translation_unit, line_window=None) -> List:
        """Extract macro definitions from Clang AST"""
        macros = []
        try:
            def visit_cursor(cursor):
                if cursor.kind == CursorKind.MACRO_DEFINITION:
                    if cursor.location.file and str(cursor.location.file) == translation_unit.spelling:
                        if line_window:
                            start_line = cursor.extent.start.line if cursor.extent else cursor.location.line
                            end_line = cursor.extent.end.line if cursor.extent else cursor.location.line
                            low = max(1, line_window[0] - line_window[2])
                            high = line_window[1] + line_window[2] if line_window[1] else float("inf")
                            if end_line < low or start_line > high:
                                return
                        macros.append(cursor)
                for child in cursor.get_children():
                    visit_cursor(child)

            if translation_unit and translation_unit.cursor:
                visit_cursor(translation_unit.cursor)
        except Exception:
            pass
        return macros

    @staticmethod
    def get_global_variables(ast, line_window=None) -> List:
        """ Collect top-level globals (VAR_DECL with parent TRANSLATION_UNIT). """
        globals_ = []
        if not ast or not ast.cursor:
            return globals_

        low = high = None
        if line_window:
            start_line, end_line, padding = line_window
            low = max(1, start_line - padding)
            high = end_line + padding if end_line else float("inf")

        def visit(node):
            if node.kind == CursorKind.VAR_DECL:
                # only keep globals defined at file scope (parent = TRANSLATION_UNIT)
                if node.semantic_parent and node.semantic_parent.kind == CursorKind.TRANSLATION_UNIT:
                    if line_window:
                        try:
                            n_start = node.extent.start.line
                            n_end = node.extent.end.line
                        except Exception:
                            n_start = n_end = node.location.line
                        if n_end < low or n_start > high:
                            return
                    globals_.append(node)
            for child in node.get_children():
                visit(child)

        visit(ast.cursor)
        return globals_
