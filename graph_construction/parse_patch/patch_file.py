
import re
from typing import List, Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .PatchHunk import PatchHunk


class PatchFile:
    def __init__(self, fname: str, isAdd: bool, isDel: bool, isMod: bool, isRename: bool):
        self.hunks: List['PatchHunk'] = []       
        self.fname = fname                     
        self.isAdd = isAdd                     
        self.isDel = isDel                      
        self.isMod = isMod                      
        self.isRename = isRename                 
    
    def addHunk(self, h: 'PatchHunk'):
        self.hunks.append(h)
        
    def getHunk(self) -> List['PatchHunk']:
        return self.hunks
        
    def __str__(self) -> str:
        if self.isAdd:
            return f"add:{self.fname}"
        elif self.isDel:
            return f"del:{self.fname}"
        elif self.isMod:
            return f"mod:{self.fname}"
        return f"rename:{self.fname}"
