
import os
from typing import List, Optional, Tuple
from clang.cindex import CursorKind
from .unit import CUnit, Position
from .clang_extract import ClangExtract
import json

class ClangGenUnits:
    """
    Core semantic unit extraction for C code.
    """

    def __init__(self, f_content: List[str], units: List[CUnit], translation_unit, line_window: Optional[Tuple[int, int, int]] = None):
        """
        Initialize ClangGenUnits and process the AST.
        """
        self.f_content = f_content  
        self.units = units or [] 
        self.ast = translation_unit
        self.line_window = line_window
        
        if self.ast is not None:
            self._gen_units()
            self._gen_code_for_units()            
            before_dedup = len(self.units)
            self._remove_dup()            
            before_filter = len(self.units)
            self._filter_empty_units()   
                     
            
    def _gen_units(self):
        """
        Extract semantic units from Clang AST.
        
        Extracted Constructs:
            - Functions, Variables, Structs, Unions, Enums, Typedefs
            - Control flow: if, for, while, do-while, switch
            - Statements: return, break, goto, label, expression
            - Preprocessor: #include, #define, macro usage
        """
        
        func_defs = ClangExtract.get_function_definitions(self.ast, self.line_window)
        for func_def in func_defs:
            self._create_unit_from_cursor(func_def, "FunctionDeclaration")

        declarations = ClangExtract.get_declarations(self.ast, self.line_window)
        for decl in declarations:
            self._create_unit_from_cursor(decl, "VariableDeclaration")
        
        struct_decls = ClangExtract.get_struct_declarations(self.ast, self.line_window)
        for s in struct_decls:
            for ch in s.get_children():
                if ch.kind == CursorKind.FIELD_DECL and ch.spelling:
                    self._create_unit_from_cursor(ch, "FieldDeclaration")

        union_decls = ClangExtract.get_union_declarations(self.ast, self.line_window)
        for udecl in union_decls:
            for ch in udecl.get_children():
                if ch.kind == CursorKind.FIELD_DECL and ch.spelling:
                    self._create_unit_from_cursor(ch, "FieldDeclaration")

        enum_decls = ClangExtract.get_enum_declarations(self.ast, self.line_window)
        for e in enum_decls:
            self._create_unit_from_cursor(e, "EnumDeclaration")
        
        typedef_decls = ClangExtract.get_typedef_declarations(self.ast, self.line_window)
        for t in typedef_decls:
            self._create_unit_from_cursor(t, "TypedefDeclaration")
        
        
        if_stmts = ClangExtract.get_if_statements(self.ast, self.line_window)
        for i in if_stmts:
            self._create_unit_from_cursor(i, "IfStatement")
        
        else_stmts = ClangExtract.get_else_statements(self.ast, self.line_window)
        for e in else_stmts:
            self._create_unit_from_cursor(e, "ElseStatement")
        
        while_stmts = ClangExtract.get_while_statements(self.ast, self.line_window)
        for w in while_stmts:
            self._create_unit_from_cursor(w, "WhileStatement")
        
        for_stmts = ClangExtract.get_for_statements(self.ast, self.line_window)
        for f in for_stmts:
            self._create_unit_from_cursor(f, "ForStatement")
        
        do_whiles = ClangExtract.get_do_while_statements(self.ast, self.line_window)
        for d in do_whiles:
            self._create_unit_from_cursor(d, "DoWhileStatement")
        
        switch_stmts = ClangExtract.get_switch_statements(self.ast, self.line_window)
        for s in switch_stmts:
            self._create_unit_from_cursor(s, "SwitchStatement")
        
        
        decl_stmts = ClangExtract.get_decl_statements(self.ast, self.line_window)
        for ds in decl_stmts:
            self._create_unit_from_cursor(ds, "VariableDeclaration")
        
        return_stmts = ClangExtract.get_return_statements(self.ast, self.line_window)
        for r in return_stmts:
            self._create_unit_from_cursor(r, "ReturnStatement")
        
        breaks = ClangExtract.get_break_statements(self.ast, self.line_window)
        for b in breaks:
            self._create_unit_from_cursor(b, "BreakStatement")
        
        goto_stmts = ClangExtract.get_goto_statements(self.ast, self.line_window)
        for g in goto_stmts:
            self._create_unit_from_cursor(g, "GotoStatement")
        
        label_stmts = ClangExtract.get_label_statements(self.ast, self.line_window)
        for l in label_stmts:
            self._create_unit_from_cursor(l, "LabelStatement")
        
        expr_stmts = ClangExtract.get_expression_statements(self.ast, self.line_window)
        for e in expr_stmts:
            # Skip empty or whitespace-only expression statements
            if e.extent and e.extent.start and e.extent.end:
                start_line = e.extent.start.line
                end_line = e.extent.end.line
                if start_line <= len(self.f_content):
                    # Check if the line content is empty or whitespace-only
                    line_content = self.f_content[start_line - 1].strip()
                    if not line_content:
                        continue
            self._create_unit_from_cursor(e, "ExpressionStatement")
        
        switch_entries = ClangExtract.get_switch_entries(self.ast, self.line_window)
        for se in switch_entries:
            self._create_unit_from_cursor(se, "SwitchEntry")
        
        
        includes = ClangExtract.get_includes(self.ast)
        for typ, sl, sc, el, ec, text in includes:
            start = Position(sl, sc)
            end   = Position(el, ec)
            u = CUnit(start, end, typ, ';')
            u.set_not_skip(True)
            u.set_code(text.strip())
            self.units.append(u)

        macros = ClangExtract.get_macro_definitions(self.ast, self.line_window)
        for m in macros:
            self._create_unit_from_cursor(m, "MacroDefinition")

        macro_uses = ClangExtract.get_macro_instantiations(self.ast, self.line_window)
        for mu in macro_uses:
            self._create_unit_from_cursor(mu, "MacroUse")

        macro_conds = ClangExtract.get_macro_conditionals(self.ast, self.line_window)
        for cond in macro_conds:
            self._create_unit_from_cursor(cond, cond.kind.name)
        
        globals_ = ClangExtract.get_global_variables(self.ast, self.line_window)
        for g in globals_:
            self._create_unit_from_cursor(g, "VariableDeclaration")
        
        

    def _create_unit_from_cursor(self, cursor, unit_type: str):
        """
        Create a CUnit from a Clang cursor node.
        1. Extract position from AST node
        2. Create Unit object
        3. Set skip flags based on node type
        4. Add to units list
        
        """
        try:
            # filter out cursors from other files (headers, includes, etc.)
            main_file = os.path.abspath(self.ast.spelling) if self.ast and self.ast.spelling else None
            if main_file:
                try:
                    loc_file = cursor.location.file
                    if loc_file is not None:
                        if os.path.abspath(loc_file.name) != main_file:
                            return
                    elif cursor.extent is not None and cursor.extent.start.file is not None:
                        if os.path.abspath(cursor.extent.start.file.name) != main_file:
                            return
                except Exception:
                    pass

            # extract position from Clang cursor
            if cursor.extent is not None:
                start_line = cursor.extent.start.line
                start_col  = cursor.extent.start.column
                end_line   = cursor.extent.end.line
                end_col    = cursor.extent.end.column
            else:
                start_line = cursor.location.line
                start_col  = cursor.location.column
                end_line   = start_line
                end_col    = start_col + (len(cursor.spelling) if cursor.spelling else 1)

            start_pos = Position(start_line, start_col)
            end_pos   = Position(end_line, end_col)

            unit = CUnit(start_pos, end_pos, unit_type, ';')
            unit.cursor = cursor
            t = unit_type

            if t in ("FunctionDeclaration", "FunctionDef"):
                unit.set_begin_char('(')
                unit.end_char = ')'
                unit.set_skip_ahead(True)
                unit.set_not_skip(False)
                unit.set_skip_two_sides(False)

            elif t in ("IfStatement", "ForStatement", "WhileStatement", 
                      "DoWhileStatement", "SwitchStatement"):
                # Find the closing paren of the condition
                closing_paren_pos = self._find_closing_paren_after_keyword(start_pos, t)
                if closing_paren_pos:
                    # Trim extent to end at closing paren
                    end_pos = closing_paren_pos
                    unit.set_end_pos(end_pos)
                
                unit.set_begin_char('(')
                unit.end_char = ')'
                unit.set_skip_two_sides(True)
                unit.set_skip_ahead(False)
                unit.set_not_skip(False)

            elif t == "ElseStatement":
                else_keyword_pos = self._find_else_keyword_before(start_pos)
                if else_keyword_pos:
                    unit.set_start_pos(else_keyword_pos)
                    unit.set_end_pos(Position(else_keyword_pos.line, else_keyword_pos.column + 4))
                    unit.end_char = 'e'
                    unit.set_not_skip(True)
                    unit.set_skip_ahead(False)
                    unit.set_skip_two_sides(False)
                else:
                    return

            elif t in ("ReturnStatement", "BreakStatement", "ContinueStatement",
                      "GotoStatement", "LabelStatement", "ExpressionStatement"):
                unit.end_char = ';'
                unit.set_not_skip(True)
                unit.set_skip_ahead(False)
                unit.set_skip_two_sides(False)

            elif t in ("VariableDeclaration"):
                
                unit.end_char = ';'
                unit.set_not_skip(True)
                unit.set_skip_ahead(False)
                unit.set_skip_two_sides(False)
                
                # Extract variable names
                if cursor.kind == CursorKind.VAR_DECL:
                    if cursor.spelling:
                        unit.add_declared_var(cursor.spelling)
                elif cursor.kind == CursorKind.DECL_STMT:
                    # DECL_STMT contains VAR_DECL children
                    for child in cursor.get_children():
                        if child.kind == CursorKind.VAR_DECL and child.spelling:
                            unit.add_declared_var(child.spelling)
                else:
                    # Check all children
                    for child in cursor.get_children():
                        if child.kind == CursorKind.VAR_DECL and child.spelling:
                            unit.add_declared_var(child.spelling)
               
            elif t in ("TypedefDeclaration"):
                unit.end_char = ';'
                unit.set_not_skip(True)
                unit.set_skip_ahead(False)
                unit.set_skip_two_sides(False)

                if cursor.spelling:
                    unit.add_declared_var(cursor.spelling)
            
            elif t in ("EnumDeclaration"):
                unit.end_char = '}'
                unit.set_not_skip(True)
                unit.set_skip_ahead(False)
                unit.set_skip_two_sides(False)

                # Extract enum constants
                for ch in cursor.get_children():
                    if ch.kind == CursorKind.ENUM_CONSTANT_DECL:
                        unit.add_declared_var(ch.spelling)

            elif t in ("StructDeclaration", "UnionDeclaration"):
                unit.end_char = '}'
                unit.set_not_skip(True)
                unit.set_skip_ahead(False)
                unit.set_skip_two_sides(False)

                # Extract field names
                for ch in cursor.get_children():
                    if ch.kind == CursorKind.FIELD_DECL:
                        unit.add_declared_field(ch.spelling)

            elif t in ("IncludeDirective", "MacroDefinition", "MacroUse"):
                # Expand to full line
                start_pos = Position(start_line, 1)
                # Find end of line in file content
                if start_line <= len(self.f_content):
                    line_content = self.f_content[start_line - 1]
                    end_col = len(line_content) + 1
                    end_pos = Position(start_line, end_col)
                
                unit.set_start_pos(start_pos)
                unit.set_end_pos(end_pos)
                unit.end_char = ';'          
                unit.set_not_skip(True)
                unit.set_skip_ahead(False)
                unit.set_skip_two_sides(False)

            elif t in ("FieldDeclaration"):
                start_pos = Position(start_line, 1)
                end_pos_extended = self._find_next_semicolon(end_pos)
                if end_pos_extended:
                    end_pos = end_pos_extended
                
                unit.set_start_pos(start_pos)
                unit.set_end_pos(end_pos)
                unit.end_char = ';'
                unit.set_not_skip(True)
                unit.set_skip_ahead(False)
                unit.set_skip_two_sides(False)
                
                if cursor.spelling:
                    unit.add_declared_field(cursor.spelling)
            
            elif t in ("SwitchEntry"):
                unit.end_char = ':'
                unit.set_not_skip(True)
                unit.set_skip_ahead(False)
                unit.set_skip_two_sides(False)

            else:
                unit.end_char = ';'
                unit.set_not_skip(True)
                unit.set_skip_ahead(False)
                unit.set_skip_two_sides(False)

            self.units.append(unit)

        except Exception as e:
            print(f"Warning: Could not create unit from cursor {cursor.kind}: {e}")

    
    def _skip_ahead(self, pos: Position) -> Position:
        """
        Move position forward by 1 character.
        """
        if pos.column == len(self.f_content[pos.line - 1]) or len(self.f_content[pos.line - 1]) == 0:
            new_pos = Position(pos.line + 1, 1)
        else:
            new_pos = Position(pos.line, pos.column + 1)
        if not self._is_valid(new_pos):
            raise Exception(f"Invalid skipAhead pos: {new_pos}")
        return new_pos
    
    def _skip_back(self, pos: Position) -> Position:
        """
        Move position backward by 1 character.
        """
        if pos.line <= 1 and pos.column <= 1:
            return pos
        if pos.column == 1 or len(self.f_content[pos.line - 1]) == 0:
            prev_len = len(self.f_content[pos.line - 2]) if pos.line > 1 else 1
            return Position(max(1, pos.line - 1), max(1, prev_len))
        return Position(pos.line, pos.column - 1)
    
    def _skip_ahead_to(self, pos: Position, char: str) -> Position:
        """
        Used for finding delimiters like '(' or ')' when extracting
        function signatures or conditions.
        """
        current_pos = Position(pos.line, pos.column)
        max_line = len(self.f_content)
        while current_pos.line <= max_line:
            # Skip empty lines
            while (current_pos.line <= max_line and 
                   len(self.f_content[current_pos.line - 1]) == 0):
                current_pos = self._skip_ahead(current_pos)
            if current_pos.line > max_line:
                break
            if self._get_char(current_pos) == char:
                return current_pos
            current_pos = self._skip_ahead(current_pos)
        return current_pos
    
    def _skip_back_to(self, pos: Position, char: str) -> Position:
        """
        Used for finding delimiters like ')' when extracting conditions from the end position.
        """
        current = Position(pos.line, pos.column)
        guard = 0
        while True:
            if current.line < 1:
                return current
            line = self.f_content[current.line - 1] if 0 < current.line <= len(self.f_content) else ""
            if line and 1 <= current.column <= len(line) and line[current.column - 1] == char:
                return current
            prev = current
            current = self._skip_back(current)
            # Safety: detect stuck position
            if current.line == prev.line and current.column == prev.column:
                return current
            guard += 1
            if guard > 1_000_000:
                return current
    
    def _find_next_semicolon(self, pos: Position) -> Position:
        """
        Find the next semicolon from the given position. Used to extend variable declaration extents to include the full statement.
        """
        current = Position(pos.line, pos.column)
        max_lines = 10  # Reasonable limit for multi-line declarations
        lines_searched = 0
        
        while current.line <= len(self.f_content) and lines_searched < max_lines:
            line = self.f_content[current.line - 1]
            for col in range(current.column - 1, len(line)):
                if line[col] == ';':
                    return Position(current.line, col + 2)
            current = Position(current.line + 1, 1)
            lines_searched += 1
        
        return pos 
    
    def _find_next_semicolon(self, pos: Position) -> Position:
        """
        Find the next semicolon from the given position.
        """
        line_idx = pos.line - 1
        col = pos.column - 1
        
        while line_idx < len(self.f_content):
            line = self.f_content[line_idx]
            while col < len(line):
                if line[col] == ';':
                    return Position(line_idx + 1, col + 2)  # +2 to include the semicolon
                col += 1
            line_idx += 1
            col = 0
        return pos 
    
    def _remove_comments(self, code: str) -> str:
        import re
        
        # Remove multi-line comments /* ... */
        code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.DOTALL)
        
        # Remove single-line comments // ...
        lines = []
        for line in code.split('\n'):
            # Find // outside of strings
            in_string = False
            escape_next = False
            comment_start = -1
            
            for i, ch in enumerate(line):
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\':
                    escape_next = True
                    continue
                if ch == '"' and not in_string:
                    in_string = True
                    continue
                if ch == '"' and in_string:
                    in_string = False
                    continue
                if ch == '/' and i + 1 < len(line) and line[i + 1] == '/' and not in_string:
                    comment_start = i
                    break
            
            if comment_start >= 0:
                lines.append(line[:comment_start].rstrip())
            else:
                lines.append(line)
        
        return '\n'.join(lines)
    
    def _skip_to_non_empty(self, pos: Position) -> Position:
        current_pos = pos
        max_line = len(self.f_content)
        while current_pos.line <= max_line:
            line = self.f_content[current_pos.line - 1] if current_pos.line - 1 < max_line else ""
            # Skip empty lines
            if len(line) == 0:
                current_pos = self._skip_ahead(current_pos)
                continue
            # Skip spaces and tabs
            while current_pos.column <= len(line) and line[current_pos.column - 1] in [' ', '\t']:
                current_pos = self._skip_ahead(current_pos)
            return current_pos
        return current_pos
    
    def _get_char(self, pos: Position) -> str:
        if (pos.line - 1 < len(self.f_content) and 
            pos.column - 1 < len(self.f_content[pos.line - 1])):
            return self.f_content[pos.line - 1][pos.column - 1]
        return ''
    
    def _is_valid(self, pos: Position) -> bool:
        if pos.line < 1 or pos.line > len(self.f_content):
            return False
        if pos.line <= len(self.f_content):
            return pos.column >= 1 and pos.column <= len(self.f_content[pos.line - 1]) + 1
        return False
    
    
    def _gen_code_for_units(self):
        
        for unit in self.units:
            try:
                if unit.get_not_skip():
                    self._get_code(unit)
                    
                elif unit.get_skip_two_sides():
                    p1 = unit.get_start_pos()
                    
                    # Find closing ')' from end
                    p2 = self._skip_back_to(unit.get_end_pos(), unit.get_end_char())
                    if self._get_char(p2) != ')':
                        p2 = self._skip_back(p2)
                    
                    unit.set_start_pos(p1)
                    unit.set_end_pos(p2)
                    self._get_code(unit)
                    
                elif unit.get_skip_ahead():
                    # Skip to '(' then to ')' (function sigs)
                    p1 = unit.get_start_pos()
                    if unit.get_has_annotation():
                        p1 = self._skip_ahead(p1)
                        p1 = self._skip_to_non_empty(p1)
                    p2 = self._skip_ahead_to(p1, unit.get_end_char())
                    if self._get_char(p2) != ')':
                        p2 = self._skip_back(p2)
                    unit.set_start_pos(p1)
                    unit.set_end_pos(p2)
                    self._get_code(unit)
                    
                else:
                    # Skip back to end_char (rare case)
                    p1 = unit.get_start_pos()
                    if unit.get_has_annotation():
                        p1 = self._skip_ahead(p1)
                        p1 = self._skip_to_non_empty(p1)
                    p2 = unit.get_end_pos()
                    p2 = self._skip_back_to(p2, unit.get_end_char())
                    if self._get_char(p2) != ')':
                        p2 = self._skip_back(p2)
                    unit.set_start_pos(p1)
                    unit.set_end_pos(p2)
                    self._get_code(unit)
                    
            except Exception as e:
                print(f"[WARN] Exception while handling {unit.get_type()} "
                      f"between {unit.get_start_pos()} and {unit.get_end_pos()}: {e}")
    
   
    
    def _get_code(self, unit: CUnit):
        start_pos = unit.get_start_pos()
        end_pos = unit.get_end_pos()

        unit_type = unit.get_type() if hasattr(unit, 'get_type') else None
        skip_cursor_extent = unit_type in ("IncludeDirective", "MacroDefinition", "MacroUse")
        
        if not skip_cursor_extent and hasattr(unit, "cursor") and unit.cursor is not None:
            try:
                start = unit.cursor.extent.start
                end = unit.cursor.extent.end
                if start.file and os.path.exists(start.file.name):
                    with open(start.file.name, "r", encoding="utf-8", errors="ignore") as f:
                        src = f.read()
                    start_offset = start.offset
                    end_offset = end.offset
                    code = src[start_offset:end_offset]
                    # Remove comments and trim whitespace
                    code = self._remove_comments(code)
                    unit.set_code(code.strip())
                    return
            except Exception as e:
                print(f"[WARN] Clang extent extraction failed for {unit.get_type()}: {e}")

        if start_pos.line == end_pos.line:
            if start_pos.line <= len(self.f_content):
                line = self.f_content[start_pos.line - 1]
                if start_pos.column <= len(line) and end_pos.column <= len(line) + 1:
                    code = line[start_pos.column - 1:end_pos.column]
                    # Remove comments
                    code = self._remove_comments(code)
                    unit.set_code(code.strip())
                    return

        code_lines = []
        for line_num in range(start_pos.line, end_pos.line + 1):
            if line_num <= len(self.f_content):
                line = self.f_content[line_num - 1]
                if line_num == start_pos.line:
                    code_lines.append(line[start_pos.column - 1:])
                elif line_num == end_pos.line:
                    code_lines.append(line[:end_pos.column])
                else:
                    code_lines.append(line)
        
        code = '\n'.join(code_lines)
        code = self._remove_comments(code)
        code = code.strip()
        
        if not code:
            unit.set_code("")  # Set empty code but don't skip the unit
            return
            
        unit.set_code(code)


    def _remove_dup(self):
        """
        Remove duplicate units based on (type, position, code).
        """
        seen = set()
        unique_units = []
        for unit in self.units:
            key = (
                unit.get_type(),
                unit.get_start_pos().line, unit.get_start_pos().column,
                unit.get_end_pos().line, unit.get_end_pos().column,
                unit.get_code_str().strip() if unit.get_code_str() else ""
            )
            if key not in seen:
                seen.add(key)
                unique_units.append(unit)
        self.units = unique_units
    
    def _filter_empty_units(self):
        filtered_units = []
        for unit in self.units:
            code = unit.get_code_str()
            if code and code.strip(): 
                filtered_units.append(unit)
        self.units = filtered_units
    
    def _find_else_keyword_before(self, pos: Position):
        import re
        
        # Search backwards from the given position
        line_idx = pos.line - 1
        col = pos.column - 1
        
        # Search up to 10 lines before
        max_search_lines = 10
        lines_searched = 0
        
        while line_idx >= 0 and lines_searched < max_search_lines:
            line = self.f_content[line_idx]
            
            search_end = col if line_idx == pos.line - 1 else len(line)
            search_text = line[:search_end]
            
            matches = list(re.finditer(r'\belse\b', search_text))
            if matches:
                last_match = matches[-1]  
                return Position(line_idx + 1, last_match.start() + 1)
            line_idx -= 1
            lines_searched += 1
            col = len(self.f_content[line_idx]) if line_idx >= 0 else 0
        
        return None
    
    def _find_closing_paren_after_keyword(self, pos: Position, stmt_type: str):
        line_idx = pos.line - 1
        col = pos.column - 1
        found_open = False
        depth = 0
        
        while line_idx < len(self.f_content) and line_idx < pos.line + 5:
            line = self.f_content[line_idx]
            while col < len(line):
                if line[col] == '(':
                    found_open = True
                    depth += 1
                elif line[col] == ')' and found_open:
                    depth -= 1
                    if depth == 0:
                        return Position(line_idx + 1, col + 1)
                col += 1
            line_idx += 1
            col = 0
        return None

    def get_all_units(self) -> List[CUnit]:
        return self.units
