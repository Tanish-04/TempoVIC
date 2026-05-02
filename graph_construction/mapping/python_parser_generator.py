import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_loader import get_gumtree_bin_path, get_gumtree_java_home
from clang import cindex
from .python_parser_visitor import CParserVisitor


class LineReader:
    def __init__(self, content: str):
        self.content = content
        self.line_starts = []
        off = 0
        for line in content.splitlines(True):
            self.line_starts.append(off)
            off += len(line)

    def position_for(self, line: int, column: int) -> int:
        if line <= 0:
            return 0
        if line - 1 >= len(self.line_starts):
            return len(self.content)
        base = self.line_starts[line - 1]
        return min(base + (column - 1), len(self.content))

class TreeContext:
    class Tree:
        def __init__(self, type_name: str, label: str = ""):
            self.type_name = type_name
            self.label = label
            self.pos = -1
            self.length = 0
            self.parent = None
            self.children = []

        def set_pos(self, pos): self.pos = pos
        def set_length(self, length): self.length = length
        def add_child(self, child):
            child.parent = self
            self.children.append(child)
        def getPos(self): return self.pos
        def getEndPos(self): return self.pos + max(self.length - 1, 0)
        def getChildren(self): return self.children
        class _T: 
            def __init__(self, name): self.name = name
        def getType(self): return TreeContext.Tree._T(self.type_name)
        def __repr__(self):
            return f"Tree(type={self.type_name}, label={self.label}, pos={self.pos}, len={self.length})"

    def __init__(self):
        self._root = None
    def create_tree(self, type_name, label=""): return TreeContext.Tree(type_name, label)
    def set_root(self, t): self._root = t
    def get_root(self): return self._root

    def to_dict(self, node=None):
        if node is None:
            node = self._root
        return {
            "type": node.type_name,
            "label": node.label,
            "pos": node.pos,
            "length": node.length,
            "children": [self.to_dict(c) for c in node.children]
        }

class CParserGenerator:
    def __init__(self, clang_library_file=None):
        import subprocess
        import tempfile
        import xml.etree.ElementTree as ET
        self.subprocess = subprocess
        self.tempfile = tempfile
        self.ET = ET
        self.gumtree_cmd = get_gumtree_bin_path()   # ← from config, not hardcoded

    def generate_from_string(self, content: str, clang_args=None) -> TreeContext:
        import os
        with self.tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(content)
            temp_file = f.name
        try:
            env = os.environ.copy()
            env['JAVA_HOME'] = get_gumtree_java_home()   # ← from config, not hardcoded
            env['PATH'] = f"{env['JAVA_HOME']}/bin:{env['PATH']}"
            result = self.subprocess.run(
                [str(self.gumtree_cmd), 'parse', temp_file],
                capture_output=True, text=True, timeout=30, env=env
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"GumTree srcML parsing failed: {result.stderr}")
            
            # Parse text output (GumTree parse outputs tree in text format)
            lr = LineReader(content)
            context = TreeContext()
            
            # Build tree from GumTree's text output
            self._build_from_text(result.stdout, context, lr)
            return context
            
        finally:
            os.unlink(temp_file)
    
    def _build_from_text(self, text_output: str, context: TreeContext, lr: LineReader):
        """Parse GumTree text format and build TreeContext"""
        import re
        
        lines = text_output.strip().split('\n')
        if not lines:
            return
        
        # Stack to track parent nodes (indent_level, tree_node)
        stack = []
        
        for line in lines:
            # Calculate indent level (number of leading spaces / 4)
            indent = len(line) - len(line.lstrip())
            indent_level = indent // 4
            
            # Parse node: "type: label [pos,endpos]" or "type [pos,endpos]"
            stripped = line.strip()
            
            # Extract position [pos,endpos]
            pos_match = re.search(r'\[(\d+),(\d+)\]$', stripped)
            if not pos_match:
                continue
            
            pos = int(pos_match.group(1))
            end_pos = int(pos_match.group(2))
            length = end_pos - pos + 1
            
            # Remove position from string
            node_str = stripped[:pos_match.start()].strip()
            
            # Parse type and label
            if ': ' in node_str:
                node_type, label = node_str.split(': ', 1)
            else:
                node_type = node_str
                label = ''
            
            # Create tree node
            tree = context.create_tree(node_type, label)
            tree.set_pos(pos)
            tree.set_length(length)
            
            # Pop stack until we find the parent for this indent level
            while stack and stack[-1][0] >= indent_level:
                stack.pop()
            
            if not stack:
                # This is the root - only set it once!
                if context.get_root() is None:
                    context.set_root(tree)
                else:
                    # Already have a root, this must be a sibling at level 0
                    # Add it as a child of the root
                    context.get_root().add_child(tree)
            else:
                # Add as child to parent
                parent_tree = stack[-1][1]
                parent_tree.add_child(tree)
            
            # Push this node onto stack
            stack.append((indent_level, tree))
