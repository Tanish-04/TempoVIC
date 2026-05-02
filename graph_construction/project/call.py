"""
Represents function calls in C code.
"""

from typing import Optional

class Call:
    
    def __init__(self, cname: str, pNum: int, fname: str, n: Optional['Node'] = None, line_number: int = 0):
        """        
        Args:
            cname: Function name being called
            pNum: Number of parameters in the call
            fname: File name where call occurs
            n: Graph node representing this call
            line_number: Line number of the call (C-specific helper)
        """
        # EXACT Java field names and types  
        self.cname = cname                              # public final String cname (function name)
        self.pNum = pNum                               # public final int pNum  
        self.fname = fname                             # public final String fname (file name)
        self.n = n                                     # public final Node n
        
        # C-specific helper field (not in Java)
        self.line_number = line_number
    
    def __str__(self) -> str:
        return f"call:{self.cname},in class {self.fname} with {self.pNum} parameters"
    
  