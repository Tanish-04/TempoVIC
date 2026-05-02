#!/usr/bin/env python3
"""
Generate Mini Graphs for NeuralSZZ baseline.

Reads graph_bszz.json files and generates mini subgraphs
centered around deleted nodes for training.

Uses composite keys "project/test_name" so downstream scripts
can resolve paths back to graph_construction/data/.

Usage:
    python genMiniGraphs.py --projects linux
    python genMiniGraphs.py --projects linux FFmpeg
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer

from config import GRAPH_DATA_DIR, NEURALSZZ_DATA_DIR, get_available_projects

tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base-nine")


def getAllGraph(projects: list) -> Dict[str, List[Dict]]:
    """
    Load all graph_bszz.json files from test cases.

    Keys are composite "project/test_name" so eval.py can resolve paths.
    """
    fDirMap = {}
    skipped = 0
    loaded = 0

    for project in projects:
        project_dir = GRAPH_DATA_DIR / project
        if not project_dir.is_dir():
            print(f"[WARN] Project directory not found: {project_dir}")
            continue

        for test_dir in sorted(project_dir.iterdir()):
            if not test_dir.is_dir():
                continue

            test_name = test_dir.name
            info_path = test_dir / "info.json"

            if not info_path.exists():
                skipped += 1
                continue

            with open(info_path) as f:
                info = json.load(f)

            fix_commit = info.get("fix")
            if not fix_commit:
                skipped += 1
                continue

            commit_dir = test_dir / fix_commit
            graph_path = commit_dir / "graph_bszz.json"

            if not graph_path.exists():
                skipped += 1
                continue

            try:
                with open(graph_path) as f:
                    graph = json.load(f)

                if graph:  # Only add non-empty graphs
                    composite_key = f"{project}/{test_name}"
                    fDirMap[composite_key] = graph
                    loaded += 1
            except Exception as e:
                print(f"Error loading {graph_path}: {e}")
                skipped += 1

    print(f"Loaded {loaded} graphs, skipped {skipped}")
    return fDirMap


def toBidirectional(graph, fDir):
    """Make all edges bidirectional."""
    for node in graph:
        node["fDir"] = fDir
        index = node["nodeIndex"]

        for e in node["cfgs"]:
            if e < len(graph) and index not in graph[e]["cfgs"]:
                graph[e]["cfgs"].append(index)

        for e in node["dfgs"]:
            if e < len(graph) and index not in graph[e]["dfgs"]:
                graph[e]["dfgs"].append(index)

        for e in node["fieldParents"]:
            if e < len(graph) and index not in graph[e]["fieldParents"]:
                graph[e]["fieldParents"].append(index)

        for e in node["methodParents"]:
            if e < len(graph) and index not in graph[e]["methodParents"]:
                graph[e]["methodParents"].append(index)


def clone(node):
    """Create a deep copy of a node."""
    cnode = {}
    cnode["cfgs"] = [e for e in node["cfgs"]]
    cnode["dfgs"] = [e for e in node["dfgs"]]
    cnode["fieldParents"] = [e for e in node["fieldParents"]]
    cnode["methodParents"] = [e for e in node["methodParents"]]
    cnode["commits"] = [cid for cid in node.get("commits", [])]

    cnode["code"] = node["code"]
    cnode["fName"] = node["fName"]
    cnode["isDel"] = node["isDel"]
    cnode["lineBeg"] = node["lineBeg"]
    cnode["lineEnd"] = node["lineEnd"]
    cnode["lineMapIndex"] = node["lineMapIndex"]
    cnode["nodeIndex"] = node["nodeIndex"]
    cnode["rootcause"] = node.get("rootcause", False)
    cnode["fDir"] = node.get("fDir", "")
    return cnode


def dfs(index, depth, graph, newGraph, visited, max_depth=2, max_nodes=8):
    """DFS to collect neighborhood around a node."""
    if depth >= max_depth or (index in visited) or len(visited) >= max_nodes:
        return

    if index >= len(graph):
        return

    newGraph.append(clone(graph[index]))
    visited.add(index)
    curNode = graph[index]

    # Traverse CFG edges (limit to 3)
    for e in curNode["cfgs"][:3]:
        if e < len(graph):
            dfs(e, depth + 1, graph, newGraph, visited, max_depth, max_nodes)

    # Traverse DFG edges (limit to 1)
    for e in curNode["dfgs"][:1]:
        if e < len(graph):
            dfs(e, depth + 1, graph, newGraph, visited, max_depth, max_nodes)

    # Traverse field parents (limit to 1)
    for e in curNode["fieldParents"][:1]:
        if e < len(graph):
            dfs(e, depth + 1, graph, newGraph, visited, max_depth, max_nodes)

    # Traverse method parents (limit to 1)
    for e in curNode["methodParents"][:1]:
        if e < len(graph):
            dfs(e, depth + 1, graph, newGraph, visited, max_depth, max_nodes)

    # Traverse lineMapIndex (connects deleted to added nodes)
    if curNode["lineMapIndex"] != -1:
        dfs(curNode["lineMapIndex"], depth + 1, graph, newGraph, visited, max_depth, max_nodes)


def adjustIndex(newGraph):
    """Re-index nodes in the mini graph to have contiguous indices."""
    if not newGraph:
        return newGraph

    delIndexMap = {}
    addIndexMap = {}
    delCnt = 0
    addCnt = 0

    for node in newGraph:
        if node["isDel"]:
            delIndexMap[node["nodeIndex"]] = delCnt
            delCnt += 1
        else:
            addIndexMap[node["nodeIndex"]] = addCnt
            addCnt += 1

    for node in newGraph:
        indexMap = delIndexMap if node["isDel"] else addIndexMap

        # Update edge lists
        for edge_key in ["cfgs", "dfgs", "fieldParents", "methodParents"]:
            node[edge_key] = [indexMap[e] for e in node[edge_key] if e in indexMap]

        # Update lineMapIndex (cross-reference between del and add)
        if node["lineMapIndex"] != -1:
            if node["isDel"]:
                node["lineMapIndex"] = addIndexMap.get(node["lineMapIndex"], -1)
            else:
                node["lineMapIndex"] = delIndexMap.get(node["lineMapIndex"], -1)

        # Update node index
        node["nodeIndex"] = indexMap[node["nodeIndex"]]

    return newGraph


def genMiniGraphs(graph, fDir):
    """Generate mini subgraphs for each deleted node in the graph."""
    allGraph = []
    toBidirectional(graph, fDir)

    for node in graph:
        if not node["isDel"]:
            continue

        node["fDir"] = fDir
        newGraph = []
        visited = set()

        dfs(node["nodeIndex"], 0, graph, newGraph, visited)

        if newGraph:
            allGraph.append(adjustIndex(newGraph))

    return allGraph


def getAllMiniGraphs(fDirMap):
    """Generate mini graphs for all test cases."""
    allMiniGraphs = {}

    for fDir, graph in fDirMap.items():
        miniGraphs = genMiniGraphs(graph, fDir)
        allMiniGraphs[fDir] = miniGraphs

    return allMiniGraphs


def genAllMiniGraphs(projects: list, output_path: str = None):
    """Main function to generate all mini graphs."""
    NEURALSZZ_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = str(NEURALSZZ_DATA_DIR / "miniGraphs_bszz.json")

    print("Loading graphs...")
    fDirMap = getAllGraph(projects)

    print("Generating mini graphs...")
    allMiniGraphs = getAllMiniGraphs(fDirMap)

    print("Tokenizing code...")
    total_minigraphs = 0
    for fDir, miniGraphs in allMiniGraphs.items():
        total_minigraphs += len(miniGraphs)
        for minig in miniGraphs:
            for node in minig:
                node["token_ids"] = tokenizer.encode_plus(
                    text=node["code"],
                    add_special_tokens=True,
                    max_length=64,
                    padding="max_length",
                )["input_ids"]

    print(f"Total test cases with mini graphs: {len(allMiniGraphs)}")
    print(f"Total mini graphs: {total_minigraphs}")

    # Also save the list of test case keys for train.py
    test_cases_path = str(NEURALSZZ_DATA_DIR / "test_cases.json")
    with open(test_cases_path, "w") as f:
        json.dump(list(allMiniGraphs.keys()), f, indent=2)
    print(f"Saved test case list to: {test_cases_path}")

    print(f"Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(allMiniGraphs, f)

    print(f"Done! Saved mini graphs to: {output_path}")


if __name__ == "__main__":
    available = get_available_projects()

    parser = argparse.ArgumentParser(
        description="Generate mini graphs for NeuralSZZ training"
    )
    parser.add_argument(
        "--projects", nargs="+",
        default=available,
        help=f"Which projects to process (available: {available})"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for miniGraphs JSON (default: data/miniGraphs_bszz.json)"
    )
    args = parser.parse_args()

    genAllMiniGraphs(projects=args.projects, output_path=args.output)
