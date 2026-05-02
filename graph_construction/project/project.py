import os
import glob
from typing import List
from pathlib import Path

from parse_joerndot.joern_graph import JoernGraph
from gen_finalgraph.gen_graph import CGraphGenerator
from gen_finalgraph.node import CGraphNode
from parse_patch.patch import Patch
from parse_patch.patch_line import PatchLine
from .source_file import CSourceFile


class CProject:
    """
    Processes multiple C source files for a single project (before or after version).
    """

    def __init__(self, project_path: str, patch_parser: Patch,
                 joern_path: str, is_after_version: bool):
        self.project_path = project_path
        self.patch_parser = patch_parser
        self.joern_path = joern_path
        self.is_after_version = is_after_version

        # Results
        self.source_files: List[CSourceFile] = []
        self.final_nodes: List[CGraphNode] = []

        self.methods = []
        self.all_calls = []

        # Process all C files in the project
        self._process_project()

    # Patch filename helpers
    def _patch_key_candidates(self, file_path: str):
        """Generate candidate keys (src_core_math.c)."""
        rel = os.path.relpath(file_path, self.project_path)
        rel_unix = rel.replace(os.sep, "/")
        java_style = rel_unix.replace("/", "_")
        base = os.path.basename(file_path)
        return [java_style, rel_unix, base]

    def _lookup_patch_lines(self, file_path: str, is_after: bool):
        candidates = self._patch_key_candidates(file_path)
        getter = self.patch_parser.getAddLines if is_after else self.patch_parser.getDelLine

        for k in candidates:
            lines = getter(k)
            if lines is not None:
                return lines

        # suffix match
        all_keys = list(self.patch_parser.addMap.keys() | self.patch_parser.delMap.keys())
        for k in all_keys:
            if any(k.endswith("_" + c) or k.endswith("/" + c) or k == c for c in candidates):
                lines = getter(k)
                if lines is not None:
                    return lines

        # Progressive suffix matching for file renames/moves.
        for candidate in candidates:
            # Only operate on underscore-format names (the first candidate)
            name, dot_ext = os.path.splitext(candidate)
            if not dot_ext:
                continue
            segments = name.split("_")
            
            # Try suffixes from longest (most specific) to shortest (just filename).
            for n_segs in range(len(segments) - 1, 0, -1):
                suffix = "_".join(segments[-n_segs:]) + dot_ext
                matches = [k for k in all_keys if k.endswith(suffix) and
                           (k == suffix or k[-(len(suffix) + 1)] == "_")]
                if len(matches) == 1:
                    lines = getter(matches[0])
                    if lines is not None:
                        print(f"[PATCH-FALLBACK] Matched source '{candidate}' "
                              f"to patch key '{matches[0]}' via suffix '{suffix}'")
                        return lines
        return None

    # Main project processing
    def _process_project(self):
        if not os.path.exists(self.project_path):
            print(f"Warning: Project path does not exist: {self.project_path}")
            return

        c_files = []
        for ext in ['*.c', '*.h']:
            c_files.extend(glob.glob(os.path.join(self.project_path, "**", ext), recursive=True))

        # Exclude files in joern directories
        c_files = [f for f in c_files if '/joern/' not in f.replace('\\', '/')]
        c_files = list(set(c_files))

        node_index = 0

        for file_path in c_files:
            file_name = os.path.basename(file_path)

            patch_lines = self._lookup_patch_lines(file_path, self.is_after_version)
            if patch_lines is None:
                continue

            add_lines = [pl.lineno for pl in patch_lines if getattr(pl, "isAdd", False)]
            del_lines = [pl.lineno for pl in patch_lines if getattr(pl, "isDel", False)]
            before_after = "after" if self.is_after_version else "before"
            print(f"[PATCH-DEBUG] {before_after} {file_name}: add_lines={add_lines[:10]}{('...' if len(add_lines) > 10 else '')}, "
                  f"del_lines={del_lines[:10]}{('...' if len(del_lines) > 10 else '')}")

            source_file = CSourceFile(
                file_path=file_path,
                joern_path=self.joern_path,
                node_index_offset=node_index,
                patch_lines=patch_lines,
                file_name=file_name
            )

            self.source_files.append(source_file)

            nodes = source_file.get_final_nodes()
            node_index += len(nodes)
            self.final_nodes.extend(nodes)

            self.methods.extend(source_file.get_methods())
            self.all_calls.extend(source_file.get_calls())


        # After processing all files, wire cross-file call refs
        self._gen_call_ref()

    def _gen_call_ref(self):
        """Wire method call references."""
        
        m_index = {}
        for m in self.methods:
            m_index.setdefault((m.name, m.pNum), []).append(m)

        for c in self.all_calls:
            for m in m_index.get((c.cname, c.pNum), []):
                caller = c.n
                callee = m.firstNode
                if caller is None or callee is None:
                    continue
                if not caller.method_parents:
                    caller.method_parents.append(callee.index)
                if caller.index not in callee.method_edges:
                    callee.method_edges.append(caller.index)

    def get_final_nodes(self) -> List[CGraphNode]:
        return self.final_nodes
    
    def get_source_files(self) -> List[CSourceFile]:
        return self.source_files
    
    def get_methods(self):
        return self.methods
    
    def get_calls(self):
        return self.all_calls
