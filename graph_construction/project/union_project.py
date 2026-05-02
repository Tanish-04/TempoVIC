import os
import json
from typing import List, Dict, Optional

from parse_joerndot.joern_graph import JoernGraph
from gen_finalgraph.gen_graph import CGraphGenerator
from gen_finalgraph.node import CGraphNode
from parse_patch.patch import Patch
from .project import CProject
from mapping.line_mapping import LineMapping
from .source_file import CSourceFile


class CUnionProject:
    """
    Combines before and after versions of a project to create the final heterogeneous graph.
    """

    def __init__(self, project_dir: str):
        self.project_dir = project_dir

        # Paths
        self.before_path = os.path.join(project_dir, "before")
        self.after_path = os.path.join(project_dir, "after")

        self.before_joern_path = os.path.join(self.before_path, "joern")
        self.after_joern_path = os.path.join(self.after_path, "joern")
        self.fixing_path = os.path.join(project_dir, "fixing")

        # Find patch file
        self.patch_path = self._find_patch_file()

        # Results
        self.final_nodes: List[CGraphNode] = []

        # Process the union project
        self._process_union()

    def _find_patch_file(self) -> Optional[str]:
        """Find the patch file in the fixing directory"""
        if not os.path.exists(self.fixing_path):
            print(f"Warning: Fixing directory not found: {self.fixing_path}")
            return None

        patch_files = os.listdir(self.fixing_path)
        if patch_files:
            return os.path.join(self.fixing_path, patch_files[0])

        return None

    def _process_union(self):
        """Process both before and after versions"""
        if not self.patch_path:
            print("Error: No patch file found")
            return

        try:
            patch_parser = Patch(self.patch_path)

            print("Processing before version...")
            before_project = CProject(
                self.before_path,
                patch_parser,
                self.before_joern_path,
                is_after_version=False
            )

            print("Processing after version...")
            after_project = CProject(
                self.after_path,
                patch_parser,
                self.after_joern_path,
                is_after_version=True
            )

            # Store projects for line mapping
            self.before_project = before_project
            self.after_project = after_project

            self._combine_nodes(before_project, after_project)
            self._generate_line_mappings()

            print(f"Generated union project with {len(self.final_nodes)} total nodes")

        except Exception as e:
            print(f"Error processing union project: {e}")

    def _adjust_index(self, offset: int, nodes: List[CGraphNode]):
        """
        Shift node.index and all edge indices by `offset`.
        Mirrors Java's adjustIndex for cfg/dfg/field/method edges and parents.
        """
        for node in nodes:
            node.index += offset

            # Edges and parents — align with CGraphNode attribute names
            if hasattr(node, "cfg_edges"):
                node.cfg_edges = [e + offset for e in node.cfg_edges]
            if hasattr(node, "cfg_parents"):
                node.cfg_parents = [e + offset for e in node.cfg_parents]

            if hasattr(node, "dfg_edges"):
                node.dfg_edges = [e + offset for e in node.dfg_edges]
            if hasattr(node, "dfg_parents"):
                node.dfg_parents = [e + offset for e in node.dfg_parents]

            if hasattr(node, "field_edges"):
                node.field_edges = [e + offset for e in node.field_edges]
            if hasattr(node, "field_parents"):
                node.field_parents = [e + offset for e in node.field_parents]

            if hasattr(node, "method_edges"):
                node.method_edges = [e + offset for e in node.method_edges]
            if hasattr(node, "method_parents"):
                node.method_parents = [e + offset for e in node.method_parents]


    def _combine_nodes(self, before_project: CProject, after_project: CProject):
        """Combine nodes from before and after versions"""
        before_nodes = before_project.get_final_nodes()
        after_nodes = after_project.get_final_nodes()

        print(f"[DEBUG] Before project nodes: {len(before_nodes)}, After project nodes: {len(after_nodes)}")

        beg_index = len(before_nodes)
        self._adjust_index(beg_index, after_nodes)
        
        for n in before_nodes:
            n_ = n.clone()
            n_.is_del = True
            self.final_nodes.append(n_)

        for n in after_nodes:
            n_ = n.clone()
            n_.is_del = False
            self.final_nodes.append(n_)

    def _generate_line_mappings(self):
        """
        Generate AST-based line mappings using LineMapping class.
        """
        try:
            # Get source files from projects
            before_sources = self.before_project.get_source_files()
            after_sources = self.after_project.get_source_files()
            
            # Match source files by name
            before_by_name = {sf.file_name: sf for sf in before_sources}
            after_by_name = {sf.file_name: sf for sf in after_sources}

            for filename in before_by_name:
                if filename in after_by_name:
                    before_src = before_by_name[filename]
                    after_src = after_by_name[filename]

                    # Use LineMapping class
                    line_mapper = LineMapping(gumtree_path="")
                    line_mapper.generate_mappings(before_src, after_src)
                    
                    before_nodes = before_src.get_final_nodes()
                    after_nodes = after_src.get_final_nodes()
                    
                    # Build lookup: original_node_index -> mapping_index
                    before_mappings = {n.index: n.mapping_index for n in before_nodes}
                    after_mappings = {n.index: n.mapping_index for n in after_nodes}
                    
                    for node in self.final_nodes:
                        if node.file_name == filename.replace("/", "_"):
                            if node.is_del and node.index in before_mappings:
                                node.mapping_index = before_mappings[node.index]
                            elif not node.is_del and node.index in after_mappings:
                                node.mapping_index = after_mappings[node.index]
                    
                    # Count mapped nodes
                    mapped_count = sum(1 for n in before_nodes if n.mapping_index != -1)
                    print(f"[LineMap] {filename}: mapped {mapped_count} node pairs")
                    print(f"[LineMap] {filename}: mapped {mapped_count}/{len(before_nodes)} deletion nodes")

        except Exception as e:
            import traceback
            print(f"Warning: GumTree AST line mapping failed: {e}")
            traceback.print_exc()

    @staticmethod
    def _write_file(content: str, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def to_line_mapping_dot(self) -> str:
        """
        Only nodes with a valid mapping index are emitted.
        """
        lines = []
        lines.append("digraph LineMapping {")
        for n in self.final_nodes:
            if n.mapping_index != -1:
                code = n.unit.get_code_str().replace('"', '\\"')
                beg = n.unit.get_start_pos().line
                end = n.unit.get_end_pos().line
                lines.append(f'  {n.index} [label="{code}({beg}-{end})"];')
        for n in self.final_nodes:
            if n.mapping_index != -1:
                lines.append(f"  {n.index} -> {n.mapping_index} [color=purple];")
        lines.append("}")
        return "\n".join(lines)

    def write_json(self, output_path: str):
        """Write the final graph to JSON format"""
        try:
            json_data = []
            for node in self.final_nodes:
                # Get code and clean whitespace characters
                code = node.unit.get_code_str()
                # Replace newlines and tabs with spaces, then collapse multiple spaces
                code = code.replace('\n', ' ').replace('\t', ' ')
                while '  ' in code:
                    code = code.replace('  ', ' ')
                code = code.strip()
                
                node_data = {
                    "cfgs": getattr(node, "cfg_edges", []),
                    "code": code,
                    "dfgs": getattr(node, "dfg_edges", []),
                    "fName": node.file_name.replace("/", "_"),
                    "fieldParents": getattr(node, "field_parents", []),
                    "isDel": node.is_del,
                    "lineBeg": node.unit.get_start_pos().line,
                    "lineEnd": node.unit.get_end_pos().line,
                    "lineMapIndex": node.mapping_index,
                    "methodParents": getattr(node, "method_parents", []),
                    "nodeIndex": node.index
                }
                json_data.append(node_data)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)

            print(f"Wrote graph JSON to: {output_path}")

        except Exception as e:
            print(f"Error writing JSON: {e}")


    def get_final_nodes(self) -> List[CGraphNode]:
        return self.final_nodes