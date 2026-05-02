"""
Represents function definitions in C code.
"""

from typing import Optional
from pycparser import c_ast


class Method:
    
    def __init__(self, name: str, className: str, decl: Optional[c_ast.FuncDef], 
                 pNum: int = 0, firstNode: Optional['Node'] = None, cdecl: Optional[object] = None,
                 start_line: int = 0, end_line: int = 0):
        """
        C doesn't have constructors, so cdecl is always None.
        
        Args:
            name: Function name
            className: File name in C
            decl: Function declaration AST node
            pNum: Number of parameters
            firstNode: First graph node representing this function
            start_line: Starting line number (C-specific helper)
            end_line: Ending line number (C-specific helper)
        """
        self.name = name                                # public final String name
        self.className = className                      # public final String className (file name in C)  
        self.decl = decl                               # public final MethodDeclaration decl
        self.cdecl = cdecl                              # public final ConstructorDeclaration cdecl (None in C)
        self.pNum = pNum                               # public final int pNum
        self.firstNode = firstNode                     # public final Node firstNode
        
        # C-specific helper fields
        self.start_line = start_line
        self.end_line = end_line
    
    def __str__(self) -> str:
        return f"method:{self.name},in class {self.className} with {self.pNum} parameters"
    
    
