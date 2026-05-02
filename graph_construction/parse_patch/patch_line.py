import re
from typing import List, Dict, Optional, Set

class PatchLine:
    
    def __init__(self, line: str, isDel: bool, isAdd: bool, isBg: bool, lineno: int):
        self.line = line
        self.isDel = isDel
        self.isAdd = isAdd
        self.isBg = isBg
        self.lineno = lineno

    def __str__(self) -> str:
        if self.isDel:
            return f"-{self.lineno}:{self.line}"
        elif self.isAdd:
            return f"+{self.lineno}:{self.line}"
        return f"{self.lineno}:{self.line}"
