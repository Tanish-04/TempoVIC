"""
Run B-SZZ on each test case's fixing commit graph.json to trace
deletion lines back and find candidate inducing commits.

Requires:
  - graph.json must exist and be non-empty for the fixing commit

Output per test case:
  <test_dir>/<fix_sha>/graph_bszz.json

Usage:
    python run_bszz.py --projects linux
    python run_bszz.py --projects linux FFmpeg
"""

import os
import sys
import json
import argparse
import subprocess
import traceback

from config import (
    SZZ_DIR, REPOS_DIR, GRAPH_DATA_DIR, PROJECT_TO_REPO,
    get_available_projects,
)

sys.path.insert(0, SZZ_DIR)

from szz.b_szz import BaseSZZ
from szz.core.abstract_szz import ImpactedFile


def file_exists_in_commit(repo_dir: str, commit: str, path: str) -> bool:
    """Return True if path exists in the tree of commit."""
    try:
        r = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}:{path}"],
            cwd=repo_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return r.returncode == 0
    except Exception:
        return False


def resolve_file_path(repo_dir: str, commit: str, underscore_name: str) -> str:
    """Convert underscore-format filename back to real git path."""
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit],
            cwd=repo_dir,
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None
        for git_path in result.stdout.strip().split('\n'):
            if git_path.endswith('.c') or git_path.endswith('.h'):
                if git_path.replace('/', '_') == underscore_name:
                    return git_path
        return None
    except Exception:
        return None


def run_bszz_node(node: dict, fix_sha: str, fname: str,
                  szz: BaseSZZ, repo_dir: str,
                  ground_truth: list) -> dict:
    """Run B-SZZ on a single deletion node. Mutates and returns node."""
    beg_line  = node["lineBeg"]
    end_line  = node["lineEnd"]
    mod_lines = list(range(beg_line, end_line + 1))
    impacted  = ImpactedFile(fname, mod_lines)

    node["rootcause"] = False
    node["commits"]   = []

    parent = f"{fix_sha}^"
    if not file_exists_in_commit(repo_dir, parent, fname):
        print(f"    [SKIP node] {fname}:{beg_line}-{end_line} (file not in parent)")
        node["commits"] = ["FILE_NOT_FOUND"]
        return node

    bug_commits = szz.find_bic(fix_sha, [impacted])
    bug_cids    = [c.hexsha for c in bug_commits]
    node["commits"] = bug_cids

    for cid in bug_cids:
        for gt in ground_truth:
            if cid.startswith(gt) or gt.startswith(cid):
                node["rootcause"] = True
                print(f"    ✓ Match: {cid[:12]}")
                break
        if node["rootcause"]:
            break

    print(f"    {fname}:{beg_line}-{end_line} → "
          f"{len(bug_cids)} commit(s)  "
          f"rootcause={node['rootcause']}")
    return node


def run_bszz_on_test(test_dir: str, test_name: str, repo_name: str) -> bool:
    """Run B-SZZ on the fixing commit graph for a single test case."""
    info_path = os.path.join(test_dir, "info.json")
    if not os.path.exists(info_path):
        print(f"    [SKIP] missing info.json")
        return False

    with open(info_path) as fh:
        info = json.load(fh)

    fix_sha = info.get("fix")
    if not fix_sha:
        print(f"    [SKIP] no 'fix' key in info.json")
        return False

    commit_dir = os.path.join(test_dir, fix_sha)
    graph_path = os.path.join(commit_dir, "graph.json")

    if not os.path.exists(graph_path):
        print(f"    [SKIP] graph.json missing")
        return False

    with open(graph_path) as fh:
        graph = json.load(fh)

    if len(graph) == 0:
        print(f"    [SKIP] graph.json is empty (0 nodes)")
        return False

    repo_dir      = os.path.join(REPOS_DIR, repo_name)
    ground_truth  = info.get("induce", [])
    output_path   = os.path.join(commit_dir, "graph_bszz.json")

    szz = BaseSZZ(repo_name, None, REPOS_DIR)

    final_nodes     = []
    found_rootcause = False
    del_count = add_count = 0

    for node in graph:
        fname = resolve_file_path(repo_dir, fix_sha, node["fName"])
        if not fname:
            fname = node["fName"].replace("_", "/")

        if node["isDel"]:
            try:
                node = run_bszz_node(
                    node, fix_sha, fname, szz, repo_dir, ground_truth)

                if node.get("rootcause"):
                    found_rootcause = True

            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"    [ERROR] {fname}:{node['lineBeg']}-{node['lineEnd']}: {e}")
                node["commits"] = ["ERROR"]

            del_count += 1
        else:
            add_count += 1

        final_nodes.append(node)

    with open(output_path, "w") as fh:
        json.dump(final_nodes, fh, indent=2)

    print(f"    del={del_count} add={add_count} "
          f"rootcause={found_rootcause} → graph_bszz.json")
    return True


def run(projects: list, dry_run: bool = False):
    total = ok = failed = skip = rootcause_found = 0

    for project in projects:
        project_dir = str(GRAPH_DATA_DIR / project)
        repo_name   = PROJECT_TO_REPO[project]

        if not os.path.isdir(project_dir):
            print(f"\n[WARN] project dir not found: {project_dir}")
            continue

        test_names = sorted(
            d for d in os.listdir(project_dir)
            if os.path.isdir(os.path.join(project_dir, d))
        )

        print(f"\n{'='*70}")
        print(f"PROJECT: {project}  ({len(test_names)} test cases)  repo: {repo_name}")
        print(f"{'='*70}")

        proj_ok = proj_fail = proj_skip = 0

        for test_name in test_names:
            test_dir = os.path.join(project_dir, test_name)
            print(f"\n  {test_name}")

            if dry_run:
                print(f"    [DRY-RUN] would run B-SZZ")
                proj_ok += 1
                continue

            try:
                result = run_bszz_on_test(test_dir, test_name, repo_name)
                if result:
                    proj_ok += 1
                    with open(os.path.join(test_dir, "info.json")) as fh:
                        info = json.load(fh)
                    fix_sha  = info.get("fix", "")
                    out_path = os.path.join(test_dir, fix_sha, "graph_bszz.json")
                    if os.path.exists(out_path):
                        with open(out_path) as fh:
                            nodes = json.load(fh)
                        if any(n.get("rootcause") for n in nodes):
                            rootcause_found += 1
                else:
                    proj_skip += 1
            except KeyboardInterrupt:
                print("\n[INTERRUPTED]")
                raise
            except Exception as e:
                print(f"    [ERROR] {e}")
                traceback.print_exc()
                proj_fail += 1

        total  += len(test_names)
        ok     += proj_ok
        failed += proj_fail
        skip   += proj_skip

        print(f"\n  {project} summary — ok={proj_ok}  fail={proj_fail}  skip={proj_skip}")

    print(f"\n{'='*70}")
    print(f"B-SZZ COMPLETE")
    print(f"{'='*70}")
    print(f"  Total test cases  : {total}")
    print(f"  Processed (ok)    : {ok}")
    print(f"  Skipped           : {skip}")
    print(f"  Failed            : {failed}")
    print(f"  Root cause found  : {rootcause_found}")
    if ok > 0:
        print(f"  GT match rate     : {rootcause_found*100/ok:.1f}%")


if __name__ == "__main__":
    available = get_available_projects()

    parser = argparse.ArgumentParser(
        description="Run B-SZZ on fixing commit graphs"
    )
    parser.add_argument(
        "--projects", nargs="+",
        default=available,
        help=f"Which projects to process (available: {available})"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without running SZZ"
    )
    args = parser.parse_args()

    run(projects=args.projects, dry_run=args.dry_run)