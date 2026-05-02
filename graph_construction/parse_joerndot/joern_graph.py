import re
from typing import List, Optional
from .joern_node import JoernNode
from .joern_edge import JoernEdge


class JoernGraph:
    def __init__(self, dot_file_path: str):
        self.dot_file_path = dot_file_path
        self.nodes: List[JoernNode] = []
        self.edges: List[JoernEdge] = []
        self.node_map = {}
        self.fcontent = ""
        self.lines: List[str] = []
        self.root = None
        self.has_void_ret = False
        self.void_ret_id = ""
        self.beg_node = None

        # Regex patterns
        self.re1 = r'digraph\s+"(.+)"\s+\{\s+'
        self.re2 = r'"(.+)"\s+\[label\s+=\s+<(.+)>\s+]'
        self.re3 = r'\s+"(.+?)"\s+->\s+"(.+?)"\s+\[\s+label\s+=\s"(.*)"\]\s*'
        self.re4 = r'\}'
        self.re5 = r'node\s+\[.*\]\s*;\s*'

        self._parse_dot_file()

    def _parse_dot_file(self):
        try:
            with open(self.dot_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                self.fcontent = f.read()

            raw_lines = self.fcontent.split('\n')
            self.lines = self._preprocess_lines(raw_lines)
            self._parse_graph()

        except FileNotFoundError:
            print(f"Warning: Joern dot file not found: {self.dot_file_path}")
        except Exception as e:
            print(f"Error parsing dot file {self.dot_file_path}: {e}")
            import traceback
            traceback.print_exc()

    def _preprocess_lines(self, raw_lines: List[str]) -> List[str]:
        processed_lines = []
        current_line = ""
        in_multiline_edge = False

        for line in raw_lines:
            if re.match(r'.*->.*\[\s*label\s*=\s*"[^"]*$', line):
                in_multiline_edge = True
                current_line = line
            elif in_multiline_edge:
                if line.strip().endswith('"]'):
                    current_line += " " + line.strip()
                    processed_lines.append(current_line)
                    current_line = ""
                    in_multiline_edge = False
                else:
                    current_line += " " + line.strip()
            else:
                processed_lines.append(line)

        return processed_lines

    def _get_matcher(self, pattern: str, line: str) -> Optional[re.Match]:
        match = re.search(pattern, line)
        if not match:
            raise RuntimeError(f"Pattern '{pattern}' does not match line: '{line}'")
        return match

    def _parse_graph_name(self, line: str) -> str:
        match = self._get_matcher(self.re1, line)
        return match.group(1)

    def _parse_node(self, line: str) -> JoernNode:
        match = self._get_matcher(self.re2, line)
        node_id = match.group(1)
        node_content = match.group(2)
        line_num = -1

        if ", " in node_content and "<BR/>" in node_content:
            parts = node_content.split("<BR/>", 1)
            first_part = parts[0].split(", ")
            if len(first_part) == 2:
                try:
                    line_num = int(first_part[1])
                    node_content = "(" + first_part[0] + "<BR/>" + (parts[1] if len(parts) > 1 else "") + ")"
                except ValueError:
                    node_content = "(" + node_content + ")"
            else:
                node_content = "(" + node_content + ")"
        elif not node_content.startswith("("):
            node_content = "(" + node_content + ")"

        node_content = node_content.replace(";", "")
        node_content = node_content.replace("&lt", "<")
        node_content = node_content.replace("&gt", ">")
        node_content = node_content.replace("&quot", '"')

        parts = [p.strip() for p in node_content.split(",")]
        if len(parts) >= 2:
            node_type = parts[0]
            code = ",".join(parts[1:])
        else:
            node_type = parts[0]
            code = None

        return JoernNode(node_id, node_type, code, line_num)

    def _parse_edge(self, line: str) -> Optional[JoernEdge]:
        match = self._get_matcher(self.re3, line)
        src = match.group(1)
        dst = match.group(2)

        if self.has_void_ret and (src == self.void_ret_id or dst == self.void_ret_id):
            return None

        edge_content = match.group(3)
        edge_match = self._get_matcher(r'(.+):\s+(.*)', edge_content)
        edge_type = edge_match.group(1).strip()
        content = edge_match.group(2)

        return JoernEdge(edge_type, content, src, dst)

    def _deal_ret(self, line: str):
        joern_node = self._parse_node(line)
        if "METHOD_RETURN" in line and "void" not in line:
            if self.beg_node is not None:
                self.beg_node.line_num = int(joern_node.line_num)
        self.has_void_ret = True
        self.void_ret_id = joern_node.node_id

    def _parse_graph(self):
        for line in self.lines:
            if "METHOD_RETURN" in line:
                self._deal_ret(line)

            if re.match(self.re1, line):
                continue
            elif re.match(self.re2, line):
                node = self._parse_node(line)
                if "METHOD" in line and "[label" in line:
                    self.beg_node = node
                self.add_node(node)
            elif re.match(self.re3, line):
                edge = self._parse_edge(line)
                if edge is not None:
                    self.add_edge(edge)
            elif re.match(self.re4, line):
                continue
            elif re.match(self.re5, line):
                continue
            elif line.strip() == "":
                continue
            else:
                raise Exception(line + " does not follow format rule")

    def add_node(self, node: JoernNode):
        self.node_map[node.node_id] = node
        self.nodes.append(node)

    def add_edge(self, edge: JoernEdge):
        if edge.src not in self.node_map:
            print(f"Warning: edge references unknown source node: {edge.src}")
            return
        self.node_map[edge.src].edges.append(edge)
        self.edges.append(edge)

    def get_root(self) -> str:
        if self.root is not None:
            return self.root
        s1 = set(self.node_map.keys())
        s2 = {edge.dst for edge in self.edges}
        candidates = [nid for nid in s1 if nid not in s2]
        assert len(candidates) == 1, f"Expected 1 root, found {len(candidates)}: {candidates}"
        self.root = candidates[0]
        return self.root

    def get_graph_name(self):
        if self.lines:
            graph_name = self._parse_graph_name(self.lines[0])
            return self.node_map.get(graph_name)
        return None

    def get_nodes(self) -> List[JoernNode]:
        return self.nodes

    def get_edges(self) -> List[JoernEdge]:
        return self.edges
