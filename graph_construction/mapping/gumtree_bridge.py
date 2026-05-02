import os
import subprocess
import tempfile
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET
from config_loader import (
    get_gumtree_java_home,
    get_gumtree_bin_path,
    get_gumtree_timeout,
)

class PositionMapper:
    def __init__(self, content: str):
        self.line_begs = []
        p1 = p2 = 0
        for line in content.split("\n"):
            p1 = p2
            self.line_begs.append(p1)
            p2 = p1 + len(line) + 1

    def get_line_num(self, offset: int) -> int:
        return self._upper_bound(self.line_begs, offset)

    def _upper_bound(self, vector: List[int], val: int) -> int:
        l, u = 0, len(vector) - 1
        while l < u:
            midp = l + (u - l) // 2
            if val < vector[midp]:
                u = midp
            else:
                l = midp + 1
        return l if l < len(vector) and val < vector[l] else len(vector)


class GumTreeMapping:
    def __init__(self, before_pos: int, after_pos: int, before_line: int, after_line: int):
        self.before_pos = before_pos
        self.after_pos = after_pos
        self.before_line = before_line
        self.after_line = after_line


class GumTreeJavaBridge:
    """
    Bridge to call GumTree’s matcher via the gumtree CLI.
    """

    def __init__(self):
        # Path to gumtree executable (update if needed)
        self.gumtree_cmd = get_gumtree_bin_path()

    def create_line_mappings(self, before_file: str, after_file: str) -> Dict[int, int]:
        with open(before_file, "r", encoding="utf-8", errors="ignore") as f:
            before_content = f.read()
        with open(after_file, "r", encoding="utf-8", errors="ignore") as f:
            after_content = f.read()

        before_mapper = PositionMapper(before_content)
        after_mapper = PositionMapper(after_content)

        xml_root = self._run_axmldiff(before_file, after_file)

        # Extract mappings from GumTree XML
        line_mappings: Dict[int, int] = {}
        for node in xml_root.findall(".//tree"):
            src_pos = int(node.get("pos", "-1"))
            dst_pos = int(node.get("other_pos", "-1"))
            if src_pos < 0 or dst_pos < 0:
                continue

            before_line = before_mapper.get_line_num(src_pos)
            after_line = after_mapper.get_line_num(dst_pos)
            
            # Only keep first mapping for a line to avoid noisy overwrites
            if before_line not in line_mappings:
                line_mappings[before_line] = after_line


        return line_mappings

    def _run_axmldiff(self, before_file: str, after_file: str):
        """Run GumTree axmldiff and return parsed XML root."""
        # Use Java 21 for GumTree (required for v4.0.0-beta6 with srcML support)
        env = os.environ.copy()
        env['JAVA_HOME'] = get_gumtree_java_home()
        env['PATH'] = f"{env['JAVA_HOME']}/bin:{env['PATH']}"
        
        cmd = [str(self.gumtree_cmd), "axmldiff", str(before_file), str(after_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=get_gumtree_timeout(), env=env)

        if result.returncode != 0:
            raise RuntimeError(f"GumTree failed: {result.stderr.strip()}")

        try:
            root = ET.fromstring(result.stdout)
            return root
        except ET.ParseError as e:
            raise RuntimeError(
                f"Failed to parse GumTree axmldiff XML: {e}\nOutput was:\n{result.stdout}"
            )

    def _call_gumtree_match(self, before_file: str, after_file: str) -> dict:
        # Use Java 21 for GumTree (required for v4.0.0-beta6 with srcML support)
        env = os.environ.copy()
        env['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'
        env['PATH'] = f"{env['JAVA_HOME']}/bin:{env['PATH']}"
        
        cmd = [str(self.gumtree_cmd), "axmldiff", str(before_file), str(after_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)

        if result.returncode != 0:
            raise RuntimeError(f"GumTree failed: {result.stderr.strip()}")

        try:
            root = ET.fromstring(result.stdout)
            return root
        except ET.ParseError as e:
            raise RuntimeError(f"Failed to parse GumTree axmldiff XML: {e}\nOutput was:\n{result.stdout}")



# Convenience function
def create_line_mappings_for_c_files(before_file: str, after_file: str) -> Dict[int, int]:
    bridge = GumTreeJavaBridge()
    return bridge.create_line_mappings(before_file, after_file)

