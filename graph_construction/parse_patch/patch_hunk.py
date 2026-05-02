import re
from typing import List, Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .patch_line import PatchLine


class PatchHunk:
    
    def __init__(self):
        self.patchLines: List['PatchLine'] = []
    
    def addLine(self, l: 'PatchLine'):
        self.patchLines.append(l)
    
    def getLines(self) -> List['PatchLine']:
        return self.patchLines
    
    def __str__(self) -> str:
        ret = ""
        for i in range(len(self.patchLines)):
            ret = ret + "\n" + str(self.patchLines[i])
        return ret
