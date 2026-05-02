"""
Clang-based Function and Call Tracker for building method reference edges.
Tracks function definitions and function calls using Clang AST.
"""

from typing import List, Dict
import clang.cindex
from clang.cindex import CursorKind
from .method import Method
from .call import Call


class ClangFunctionTracker:
    """
    Clang-based tracker for function definitions and calls to build method reference edges.
    """
    
    def __init__(self, file_path: str, translation_unit):
        self.file_path = file_path
        self.translation_unit = translation_unit
        self.functions = []
        self.function_calls = []
        
        # Extract functions and calls
        self._extract_functions()
        self._extract_function_calls()
    
    def _extract_functions(self):
        """Extract function definitions from Clang AST"""
        
        def visit_cursor(cursor):
            # Only process nodes from our target file (not system headers)
            if cursor.location.file and str(cursor.location.file) == self.translation_unit.spelling:
                
                # Find function definitions
                if cursor.kind == CursorKind.FUNCTION_DECL:
                    # Get function name
                    func_name = cursor.spelling if cursor.spelling else "unknown"
                    
                    # Count parameters
                    param_count = 0
                    for child in cursor.get_children():
                        if child.kind == CursorKind.PARM_DECL:
                            param_count += 1
                    
                    # Get line information from Clang
                    start_line = cursor.location.line
                    # Use extent for end line if available
                    if cursor.extent and cursor.extent.end:
                        end_line = cursor.extent.end.line
                    else:
                        end_line = start_line + 10 
                    
                    # Create Method object with EXACT same fields as FunctionTracker
                    function = Method(
                        name=func_name,
                        className=self.file_path,  
                        decl=cursor,               # Store Clang cursor 
                        pNum=param_count,        
                        firstNode=None,           
                        start_line=start_line,     # C-specific helper
                        end_line=end_line          # C-specific helper
                    )
                    
                    self.functions.append(function)
            
            # Recursively visit children
            for child in cursor.get_children():
                visit_cursor(child)
        
        # Start traversal from root cursor
        if self.translation_unit and self.translation_unit.cursor:
            visit_cursor(self.translation_unit.cursor)
    
    def _extract_function_calls(self):
        """Extract function calls from Clang AST"""
        
        def visit_cursor(cursor):
            # Only process nodes from our target file
            if cursor.location.file and str(cursor.location.file) == self.translation_unit.spelling:
                
                # Find function calls
                if cursor.kind == CursorKind.CALL_EXPR:
                    # Get function name being called (use cursor.spelling directly)
                    func_name = cursor.spelling if cursor.spelling else "unknown"
                    
                    param_count = 0
                    children = list(cursor.get_children())
                    if len(children) > 0:
                        # Count all children except the first one (function reference)
                        param_count = len(children) - 1
                    
                   
                    call = Call(
                        cname=func_name,
                        pNum=param_count,
                        fname=self.file_path,
                        n=None,
                        line_number=cursor.location.line
                    )
                    
                    self.function_calls.append(call)
            
            # Recursively visit children
            for child in cursor.get_children():
                visit_cursor(child)
        
        # Start traversal from root cursor
        if self.translation_unit and self.translation_unit.cursor:
            visit_cursor(self.translation_unit.cursor)
    
    def associate_with_nodes(self, nodes: List['Node']):
        """Associate functions and calls with graph nodes"""

        # Associate functions with their first nodes
        for function in self.functions:
            for node in nodes:
                unit_start = node.unit.get_start_pos().line
                unit_end = node.unit.get_end_pos().line
                node_code = node.unit.get_code_str()
                
                # Check line range
                line_match = unit_start <= function.start_line <= unit_end
                
                # Check name in code
                name_match = function.name.lower() in node_code.lower() if node_code else False
                
                # Check if node represents this function
                if line_match and name_match:
                    function.firstNode = node 
                    break
        
        # Associate function calls with nodes
        for call in self.function_calls:
            for node in nodes:
                unit_start = node.unit.get_start_pos().line
                unit_end = node.unit.get_end_pos().line
                node_code = node.unit.get_code_str()
                
                # Check line range
                line_match = unit_start <= call.line_number <= unit_end
                
                # Check name in code
                name_match = call.cname.lower() in node_code.lower() if node_code else False
                
                # Check if node contains this function call
                if line_match and name_match:
                    call.n = node
                    break
    
    def build_method_edges(self, nodes: List['Node']):
        """Build method reference edges between function calls and definitions"""
        # Associate nodes first
        self.associate_with_nodes(nodes)
        
        # Build edges from calls to function definitions
        for call in self.function_calls:
            if call.n is None:  
                continue
                
            # Find matching function definition
            for function in self.functions:
                if (function.firstNode is not None and
                    function.name == call.cname and
                    function.pNum == call.pNum):

                    # Add method edge from call to function
                    if function.firstNode.index not in call.n.method_edges:
                        call.n.method_edges.append(function.firstNode.index)
                    
                    # Add method parent from function to call  
                    if call.n.index not in function.firstNode.method_parents:
                        function.firstNode.method_parents.append(call.n.index)
                    
                    break
    
    def get_functions(self) -> List[Method]:
        """Get all extracted functions"""
        return self.functions
    
    def get_function_calls(self) -> List[Call]:
        """Get all extracted function calls"""
        return self.function_calls
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about functions and calls"""
        return {
            "total_functions": len(self.functions),
            "total_calls": len(self.function_calls),
            "functions_with_nodes": len([f for f in self.functions if f.firstNode is not None]),
            "calls_with_nodes": len([c for c in self.function_calls if c.n is not None])
        } 