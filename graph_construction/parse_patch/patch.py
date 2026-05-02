import re
from typing import List, Dict, Optional, Set

from .patch_file import PatchFile
from .patch_hunk import PatchHunk
from .patch_line import PatchLine

class Patch:
    
    class LineInfo:
        def __init__(self, addLineBeg: int, addLineNum: int, delLineBeg: int, delLineNum: int):
            self.addLineBeg = addLineBeg
            self.addLineNum = addLineNum  
            self.delLineBeg = delLineBeg
            self.delLineNum = delLineNum
        
        def __str__(self) -> str:
            return f"({self.addLineBeg},{self.addLineNum})({self.delLineBeg},{self.delLineNum})"
    
    def __init__(self, path: str):
        self.content = ""                                           
        self.patchFiles: List[PatchFile] = []                      
        self.lines: List[str] = []                                
        self.lptr = 0                                              
        self.commitId = ""                                         
        self.addMap: Dict[str, List[PatchLine]] = {}              # HashMap<String, ArrayList<PatchLine>> addMap
        self.delMap: Dict[str, List[PatchLine]] = {}              # HashMap<String, ArrayList<PatchLine>> delMap
        self.specialSet: Set[str] = set()                         # HashSet<String> specialSet
        
        self.fileRegex = re.compile(r"^diff(\s)+--git(\s)+a/(\S)+(\s)+b/(\S)+")          # Pattern fileRegex
        self.lineRegex = re.compile(r"^@@(\s)+-([0-9]*),([0-9]*)(\s)+\+([0-9]*),([0-9]*)(\s)+@@")  # Pattern lineRegex  
        self.endRegex = re.compile(r"--(\s)+")                                            # Pattern endRegex
        
        try:
            # Read file content
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                self.content = f.read()
            
            self.lines = self.content.split('\n')
            self.lptr = 0
            self.parse()        
            self.buildMap()    
            
        except Exception as e:
            print(f"Error processing patch file: {e}")
    
    def buildMap(self):
        for pf in self.patchFiles:
            fullName = pf.fname
            
            addPlines: List[PatchLine] = []
            delPlines: List[PatchLine] = []
            
            for h in pf.getHunk():
                for pl in h.getLines():
                    if pl.isAdd:
                        addPlines.append(pl)
                    elif pl.isDel:
                        delPlines.append(pl)
            
            self.addMap[fullName] = addPlines
            self.delMap[fullName] = delPlines
    
    def parse(self) -> bool:
        """
        Parse method
        
        Standard git patches have this structure:
        1. Commit message
        2. diff --git a/path b/path
        3. index ...
        4. --- a/path
        5. +++ b/path
        6. @@ hunks @@
        
        """
        # Skip to first diff --git line (start of actual patch content)
        while self.lptr < len(self.lines) and not self.fileRegex.search(self.lines[self.lptr]):
            self.lptr += 1
        
        # Parse all files starting from first diff --git
        self.parseFiles()        
        return True
    
    def getLineInfo(self, curLine: str) -> 'Patch.LineInfo':
        m = self.lineRegex.search(curLine)
        if not m:
            raise RuntimeError(f"can not match line:{curLine}")
        
        endPos = m.end()
        curLine = curLine[:endPos]
        
        info1 = curLine.split(" ")[1].split(",")
        info2 = curLine.split(" ")[2].split(",")
        
        # Remove the '-' prefix (substring(1))
        beg1 = int(info1[0][1:])  
        num1 = int(info1[1])
        
        # Remove the '+' prefix (substring(1))
        beg2 = int(info2[0][1:])    
        num2 = int(info2[1])
        
        return self.LineInfo(beg2, num2, beg1, num1)
    
    def parseHunk(self, ptr1: int, ptr2: int, info: 'Patch.LineInfo') -> PatchHunk:
        addNum = 0
        addLineno = info.addLineBeg
        delNum = 0  
        delLineno = info.delLineBeg
        
        h = PatchHunk()
        
        for i in range(ptr1, ptr2 + 1):
            if self.lines[i] == "-- ":
                break
            
            if self.lines[i].startswith("+"):
                addNum += 1
                h.addLine(PatchLine(self.lines[i][1:].strip(), False, True, False, addLineno))
                addLineno += 1
            elif self.lines[i].startswith("-"):
                delNum += 1
                h.addLine(PatchLine(self.lines[i], True, False, False, delLineno))
                delLineno += 1
            else:
                addLineno += 1
                delLineno += 1
                addNum += 1
                delNum += 1
            
           
        
        return h
    
    def parseFile(self, ptr1: int, ptr2: int) -> PatchFile:
        """EXACT Java parseFile method"""
        ptr = ptr1
        isAdd = False
        isDel = False
        isMod = False
        isRename = False
        
        fs = self.lines[ptr].split(" ")

        # Remove 'a/' prefix (substring(2))
        f1 = fs[len(fs) - 2][2:]  
        # Remove 'b/' prefix (substring(2))
        f2 = fs[len(fs) - 1][2:]  
        
        f1 = f1.replace("/", "_")  # replaceAll("/", "_")
        f2 = f2.replace("/", "_")  # replaceAll("/", "_")
        ptr += 1
        
        if self.lines[ptr].startswith("similarity index") and f1 != f2:
            self.specialSet.add(f1)
            self.specialSet.add(f2)
            isRename = True
        elif self.lines[ptr].startswith("deleted file mode"):
            self.specialSet.add(f1)
            self.specialSet.add(f2)
            isDel = True
        elif self.lines[ptr].startswith("new file mode"):
            self.specialSet.add(f1)
            self.specialSet.add(f2)
            isAdd = True
        else:
            isMod = True
        
        while ptr <= ptr2 and not self.lineRegex.search(self.lines[ptr]):
            ptr += 1
        
        pf = PatchFile(f2, isAdd, isDel, isMod, isRename)
        
        while ptr <= ptr2 and self.lineRegex.search(self.lines[ptr]):
            lineInfo = self.getLineInfo(self.lines[ptr])
            ptr += 1
            beg = ptr
            while ptr <= ptr2 and not self.lineRegex.search(self.lines[ptr]):
                ptr += 1
            
            pf.addHunk(self.parseHunk(beg, ptr - 1, lineInfo))
            
            if ptr > ptr2:
                break
        
        return pf
    
    def parseFiles(self):
        while self.lptr < len(self.lines) and not self.fileRegex.search(self.lines[self.lptr]):
            self.lptr += 1
        
        while self.lptr < len(self.lines) and self.fileRegex.search(self.lines[self.lptr]):
            beg = self.lptr
            self.lptr += 1
            
            # Find end of this file section
            while self.lptr < len(self.lines) and not self.fileRegex.search(self.lines[self.lptr]):
                self.lptr += 1
            
            self.patchFiles.append(self.parseFile(beg, self.lptr - 1))
            
            if self.lptr >= len(self.lines):
                break
    
    def getPatchFiles(self) -> List[PatchFile]:
        return self.patchFiles
    
    def isSpecial(self, fName: str) -> bool:
        return fName in self.specialSet
    
    def getAddLines(self, fName: str) -> Optional[List[PatchLine]]:
        return self.addMap.get(fName)
    
    def getDelLine(self, fName: str) -> Optional[List[PatchLine]]:
        return self.delMap.get(fName)