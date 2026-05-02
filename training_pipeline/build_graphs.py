"""
Builds pre-computed graph structures (nodes, edges, temporal positions)
from raw graph construction output. Supports two modes:

  full_graph    — Temporal edges connecting matched nodes across historical
                  commits (TEMPORAL_FWD). Used by the main GAT model.
  no_temporal   — Same section structure but without temporal edges.
                  Used by ablation variants (DeepSets, Transformer).

Usage:
    python build_graphs.py --mode full_graph
    python build_graphs.py --mode no_temporal
    python build_graphs.py --mode all          # builds both

Output:
    temporal_graph/full_graph/<test_name>/del_<node_idx>.json
    temporal_graph/no_temporal/<test_name>/del_<node_idx>.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

from config_utils import ConfigManager
from data_processing.phase1.processing import (
    build_full_graph_structure,
    build_sections_and_chains,
    build_cfg_dfg_edges,
    build_mini_graph,
    find_commit_dir,
    find_history_node,
    make_synthetic_node,
)

CONFIG = ConfigManager().raw
logger = logging.getLogger(__name__)

def build_no_temporal_graph_structure(
    all_nodes: List[Dict],
    del_node_idx: int,
    test_name: str,
    data_path: Path,
) -> Dict:
    (
        subgraph_nodes,
        temporal_positions,
        _section_starts,
        _temporal_chains,
        intra_section_edges,
    ) = build_sections_and_chains(all_nodes, del_node_idx, test_name, data_path)

    return {
        "nodes": subgraph_nodes,
        "edges": intra_section_edges,
        "temporal_positions": temporal_positions,
    }


MODES = {
    "full_graph": {
        "builder": build_full_graph_structure,
        "subdir": "full_graph",
    },
    "no_temporal": {
        "builder": build_no_temporal_graph_structure,
        "subdir": "no_temporal",
    },
}
def build_graphs(
    mode_name: str,
    data_path: Path,
    output_dir: Path,
    test_cases: list,
) -> int:
    """Build and save graphs for all deletion lines across test cases."""
    mode = MODES[mode_name]
    builder = mode["builder"]
    subdir = mode["subdir"]
    mode_dir = output_dir / subdir
    mode_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Mode: {mode_name}")
    print(f"Output: {mode_dir}")
    print(f"{'='*60}")

    total_saved = 0
    skipped = 0
    skipped_names: list = []
    start_time = time.time()

    for tc_idx, test_name in enumerate(test_cases):
        test_dir = data_path / test_name

        info_path = test_dir / "info.json"
        if not info_path.exists():
            skipped += 1
            skipped_names.append((test_name, "info.json not found"))
            continue

        try:
            with open(info_path) as f:
                info = json.load(f)
            fix_commit = info["fix"]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            skipped += 1
            skipped_names.append((test_name, f"info.json error: {exc}"))
            continue

        fix_commit_dir = find_commit_dir(test_dir, fix_commit)
        if not fix_commit_dir:
            skipped += 1
            skipped_names.append(
                (test_name, f"commit dir not found: {fix_commit[:12]}")
            )
            continue

        graph_path = fix_commit_dir / "graph_vszz_full_history.json"
        if not graph_path.exists():
            skipped += 1
            skipped_names.append((test_name, "vszz graph file not found"))
            continue

        try:
            with open(graph_path) as f:
                all_nodes = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            skipped += 1
            skipped_names.append((test_name, f"graph read error: {exc}"))
            continue

        test_output_dir = mode_dir / test_name
        test_output_dir.mkdir(parents=True, exist_ok=True)

        for node_idx, node in enumerate(all_nodes):
            if not node.get("isDel", False):
                continue

            structure = builder(all_nodes, node_idx, test_name, data_path)
            if structure is None:
                logger.warning(
                    "No graph structure for %s node %d", test_name, node_idx
                )
                continue

            output = {
                "test_name": test_name,
                "graph_mode": subdir,
                "node_idx_in_graph": node_idx,
                "rootcause": node.get("rootcause", False),
                "del_idx": 0,
                "nodes": structure["nodes"],
                "edges": [[s, d, t] for s, d, t in structure["edges"]],
                "temporal_positions": structure["temporal_positions"],
            }

            save_path = test_output_dir / f"del_{node_idx}.json"
            try:
                with open(save_path, "w") as f:
                    json.dump(output, f)
                total_saved += 1
            except OSError as exc:
                logger.error("Failed to save %s: %s", save_path, exc)

        if (tc_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (tc_idx + 1) / elapsed
            print(
                f"  {tc_idx + 1}/{len(test_cases)} test cases | "
                f"{total_saved} graphs saved | {rate:.1f} cases/sec"
            )

    elapsed = time.time() - start_time
    print(
        f"\nDone ({mode_name}): {total_saved} graphs saved, "
        f"{skipped} test cases skipped ({elapsed:.1f}s)"
    )
    if skipped_names:
        print(f"\nSkipped test cases:")
        for name, reason in skipped_names:
            print(f"  {name}: {reason}")
    print(f"Output: {mode_dir}")
    return total_saved



def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute graph structures for TempoVIC training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=list(MODES.keys()) + ["all"],
        help="Graph mode to build: full_graph, no_temporal, or all",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=CONFIG["paths"]["data_root"],
        help="Path to data directory containing test case folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=CONFIG["paths"]["prebuilt_dir"],
        help="Output directory for pre-built graphs",
    )
    parser.add_argument(
        "--test_cases_file",
        type=str,
        default=str(
            Path(CONFIG["paths"]["data_root"]) / CONFIG["paths"]["test_cases_file"]
        ),
        help="JSON file listing test case names",
    )

    args = parser.parse_args()

    with open(args.test_cases_file) as f:
        test_cases = json.load(f)
    print(f"Loaded {len(test_cases)} test cases")

    modes_to_build = list(MODES.keys()) if args.mode == "all" else [args.mode]

    grand_total = 0
    for mode_name in modes_to_build:
        total = build_graphs(
            mode_name,
            Path(args.data_path),
            Path(args.output_dir),
            test_cases,
        )
        grand_total += total

    print(f"\n{'='*60}")
    print(f"ALL DONE: {grand_total} total graphs saved across {len(modes_to_build)} mode(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
