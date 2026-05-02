from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class Position:
    """Represents a position in source code (line, column)"""
    line: int
    column: int
    
    def is_before(self, other: 'Position') -> bool:
        """Check if this position is before another position"""
        if self.line < other.line:
            return True
        elif self.line == other.line:
            return self.column < other.column
        return False
    
    def __eq__(self, other: 'Position') -> bool:
        return self.line == other.line and self.column == other.column


class CUnit:
    """
    Represents a semantic unit of C code (statement, declaration, expression, etc.)
    """
    
    def __init__(self, start_pos: Position, end_pos: Position, unit_type: str, end_char: str = ';'):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.unit_type = unit_type
        self.end_char = end_char
        
        # Code content and processing flags
        self.code = ""
        self.skip_ahead = False
        self.has_annotation = False
        self.skip_two_sides = False
        self.not_skip = False
        self.begin_char = ""
        
        # Associated graph node
        self.node = None
        self._declared_fields = []
        self._declared_vars = []
        self.cursor = None


    def get_declared_field_names(self):
        return list(self._declared_fields)

    def add_declared_field(self, name: str):
        if name:
            self._declared_fields.append(name)

    def get_declared_var_names(self):
        return list(self._declared_vars)

    def add_declared_var(self, name: str):
        if name:
            self._declared_vars.append(name)


    def get_start_pos(self) -> Position:
        return self.start_pos
    
    def get_end_pos(self) -> Position:
        return self.end_pos
    
    def set_start_pos(self, pos: Position):
        self.start_pos = pos
        
    def set_end_pos(self, pos: Position):
        self.end_pos = pos
    
    def get_type(self) -> str:
        return self.unit_type
    
    def get_end_char(self) -> str:
        return self.end_char
    
    def set_code(self, code_str: str):
        self.code = code_str.strip()

    
    def get_code_str(self) -> str:
        return self.code
    
    def set_skip_ahead(self, skip: bool):
        self.skip_ahead = skip
    
    def get_skip_ahead(self) -> bool:
        return self.skip_ahead
    
    def set_has_annotation(self, has: bool):
        self.has_annotation = has
    
    def get_has_annotation(self) -> bool:
        return self.has_annotation
    
    def set_skip_two_sides(self, skip: bool):
        self.skip_two_sides = skip
    
    def get_skip_two_sides(self) -> bool:
        return self.skip_two_sides
    
    def set_not_skip(self, not_skip: bool):
        self.not_skip = not_skip
    
    def get_not_skip(self) -> bool:
        return self.not_skip
    
    def set_begin_char(self, char: str):
        self.begin_char = char
    
    def get_begin_char(self) -> str:
        return self.begin_char
    
    def set_node(self, node: Any):
        self.node = node
    
    def get_node(self) -> Any:
        return self.node
    
    def is_statement(self) -> bool:
        return self.unit_type.endswith("Statement")

    def is_field_declaration(self) -> bool:
        return self.unit_type == "FieldDeclaration"

    def is_method_declaration(self) -> bool:
        return self.unit_type == "MethodDeclaration"

    def is_expression(self) -> bool:
        return self.unit_type.endswith("Expression")

    def is_declaration(self) -> bool:
        return self.unit_type.endswith("Declaration")
        
    def is_field_declaration(self) -> bool:
        return self.unit_type in ["FieldDeclaration", "StructField", "VariableDeclaration"]

    def is_function_definition(self) -> bool:
        return self.unit_type in ["FunctionDeclaration", "FunctionDef"]

    def is_struct_declaration(self) -> bool:
        return self.unit_type in ["StructDeclaration", "StructDef"]

    def is_enum_declaration(self) -> bool:
        return self.unit_type in ["EnumDeclaration"]

    def is_typedef_declaration(self) -> bool:
        return self.unit_type in ["TypedefDeclaration"]

    def __eq__(self, other: 'CUnit') -> bool:
        """Check equality based on position"""
        if not isinstance(other, CUnit):
            return False
        return (self.start_pos == other.start_pos and 
                self.end_pos == other.end_pos)
    
    def __lt__(self, other: 'CUnit') -> bool:
        """For sorting units by position"""
        if self.start_pos.is_before(other.start_pos):
            return True
        elif self.start_pos == other.start_pos:
            return self.end_pos.is_before(other.end_pos)
        return False
    
    def __str__(self) -> str:
        return f"CUnit(type={self.unit_type}, pos={self.start_pos.line}:{self.start_pos.column}-{self.end_pos.line}:{self.end_pos.column}, code='{self.code[:50]}...')"
    
    def __repr__(self) -> str:
        return self.__str__() 
