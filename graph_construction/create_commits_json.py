"""
Schema:
{
    "fix_commit":              "abc123...",
    "ground_truth":            ["def456..."],
    "vszz_introducer_commits": ["ghi789..."],   # oldest commit per chain
    "all_commits_in_history":  ["jkl012...", ...],
    "matches_ground_truth":    true/false,
    "graph_type":              "vszz_full_history",
    "stats": {
        "num_introducers":    N,
        "num_history_commits": N,
        "nodes_with_history": N
    }
}

Usage:
    python create_commits_json.py --projects linux
    python create_commits_json.py --projects FFmpeg OpenSSL PHP-SRC ImageMagick
    python create_commits_json.py --projects all
"""

import os
import json
import argparse
from typing import Dict, List, Set

from config_loader import get_all_projects
from pipeline_utils import DATA_DIR, PROJECT_TO_REPO


def is_valid_history_commit(commit: str) -> bool:
    if not commit:
        return False
    if commit in ("FILE_NOT_FOUND", "ERROR", "UNKNOWN_NEW_FILE", "UNKNOWN"):
        return False
    if len(commit) < 12:
        return False
    return True


def extract_commits_from_vszz_graph(vszz_graph: List[Dict]) -> Dict:
    commit_map: Dict[str, str] = {}   # 12-char prefix → full 40-char SHA
    raw_introducers: Set[str]  = set()
    raw_all:         Set[str]  = set()
    nodes_with_history = 0

    def collect_commit_for_set(commit: str, target: Set[str]):
        if not is_valid_history_commit(commit):
            return
        target.add(commit)
        if len(commit) == 40:
            commit_map[commit[:12]] = commit

    for node in vszz_graph:
        if not node.get("isDel", False):
            continue

        for c in node.get("commits", []):
            collect_commit_for_set(c, raw_all)

        for c in node.get("introducer_commits", []):
            collect_commit_for_set(c, raw_introducers)
            collect_commit_for_set(c, raw_all)

        chains = node.get("history_chains", [])
        if chains:
            nodes_with_history += 1
            for chain in chains:
                for entry in chain.get("history", []):
                    collect_commit_for_set(entry.get("commit", ""), raw_all)
                collect_commit_for_set(chain.get("introducer", ""), raw_introducers)

    def normalize_commit_set(raw: Set[str]) -> List[str]:
        """Deduplicate by 12-char prefix, prefer full SHAs."""
        seen: Set[str] = set()
        result: List[str] = []
        for commit in raw:
            prefix = commit[:12]
            if prefix in seen:
                continue
            seen.add(prefix)
            full = commit_map.get(prefix, commit)
            result.append(full)
        return sorted(result)

    return {
        "introducer_commits":  normalize_commit_set(raw_introducers),
        "all_history_commits": normalize_commit_set(raw_all),
        "nodes_with_history":  nodes_with_history,
    }


def create_commits_json(test_dir: str, test_name: str):
    """
    Build and write commits.json for one test case.
    Returns the commits dict on success.
    """
    info_path = os.path.join(test_dir, "info.json")
    if not os.path.exists(info_path):
        print(f"    [SKIP] missing info.json")
        return None

    with open(info_path) as fh:
        info = json.load(fh)

    fix_sha = info.get("fix")
    if not fix_sha:
        print(f"    [SKIP] no 'fix' key in info.json")
        return None

    vszz_path = os.path.join(test_dir, fix_sha, "graph_vszz.json")
    if not os.path.exists(vszz_path):
        print(f"    [SKIP] graph_vszz.json not found — run step 4 first")
        return None

    with open(vszz_path) as fh:
        vszz_graph = json.load(fh)

    ground_truth = info.get("induce", [])
    extracted    = extract_commits_from_vszz_graph(vszz_graph)

    introducers = extracted["introducer_commits"]
    all_history = extracted["all_history_commits"]

    all_candidates = set(introducers) | set(all_history)
    matches_gt = any(
        c.startswith(gt) or gt.startswith(c)
        for c in all_candidates
        for gt in ground_truth
    )

    commits_data = {
        "fix_commit":              fix_sha,
        "ground_truth":            ground_truth,
        "vszz_introducer_commits": introducers,
        "all_commits_in_history":  all_history,
        "matches_ground_truth":    matches_gt,
        "graph_type":              "vszz_full_history",
        "stats": {
            "num_introducers":     len(introducers),
            "num_history_commits": len(all_history),
            "nodes_with_history":  extracted["nodes_with_history"],
        },
    }

    commits_path = os.path.join(test_dir, "commits.json")
    with open(commits_path, "w") as fh:
        json.dump(commits_data, fh, indent=2)

    print(f"    history={len(all_history)}  "
          f"introducers={len(introducers)}  "
          f"gt_match={matches_gt}")
    return commits_data


def run(projects: list) -> None:
    total      = 0
    ok         = 0
    skip       = 0
    gt_matches = 0

    for project in projects:
        project_dir = os.path.join(DATA_DIR, project)

        if not os.path.isdir(project_dir):
            print(f"\n[WARN] project dir not found: {project_dir}")
            continue

        test_names = sorted(
            d for d in os.listdir(project_dir)
            if os.path.isdir(os.path.join(project_dir, d))
        )

        print(f"\n{'='*70}")
        print(f"PROJECT: {project}  ({len(test_names)} test cases)")
        print(f"{'='*70}")

        proj_ok = proj_skip = proj_gt = 0

        for test_name in test_names:
            test_dir = os.path.join(project_dir, test_name)
            print(f"\n  {test_name}")

            result = create_commits_json(test_dir, test_name)
            if result:
                proj_ok += 1
                if result.get("matches_ground_truth"):
                    proj_gt += 1
            else:
                proj_skip += 1

        total      += len(test_names)
        ok         += proj_ok
        skip       += proj_skip
        gt_matches += proj_gt

        print(f"\n  {project} summary — "
              f"ok={proj_ok}  skip={proj_skip}  gt_match={proj_gt}")

    print(f"\n{'='*70}")
    print(f"STEP 4 COMPLETE")
    print(f"{'='*70}")
    print(f"  Total test cases      : {total}")
    print(f"  commits.json written  : {ok}")
    print(f"  Skipped               : {skip}")
    print(f"  Ground truth matched  : {gt_matches}")
    if ok > 0:
        print(f"  GT match rate         : {gt_matches*100/ok:.1f}%")

if __name__ == "__main__":
    _all_projects = get_all_projects()

    parser = argparse.ArgumentParser(
        description="Create commits.json from graph_vszz.json"
    )
    parser.add_argument(
        "--projects", nargs="+",
        default=_all_projects,
        choices=_all_projects,
        help="Which projects to process (default: all configured projects)"
    )
    args = parser.parse_args()

    run(projects=args.projects)