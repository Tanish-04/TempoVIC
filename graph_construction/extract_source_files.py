"""
Handles source file extraction for both fix commit and history commits.

Usage:
    python extract_source_files.py --mode fix
    python extract_source_files.py --mode history
    python extract_source_files.py --mode fix --projects FFmpeg OpenSSL PHP-SRC
    python extract_source_files.py --mode history --projects FFmpeg OpenSSL PHP-SRC
    python3 extract_source_files.py --mode fix --projects linux
"""

import os
import json
import shutil
import subprocess
import traceback
import argparse

from pipeline_utils import (
    DATA_DIR,
    REPOS_DIR,
    PROJECT_TO_REPO,
    get_file_content,
    resolve_underscore_name_to_git_path,
    find_unambiguous_file_by_basename,
    commit_has_no_parent,
    get_fix_files_from_after_dir,
)

def write_commit_patch_file(commit_sha: str, fixing_dir: str, repo_dir: str):
    """Write git show output as a patch file into fixing/."""
    try:
        patch = subprocess.run(
            ["git", "show", commit_sha],
            cwd=repo_dir, capture_output=True, timeout=30
            # Note: no text=True — capture raw bytes
        )
        if patch.returncode == 0:
            patch_path = os.path.join(fixing_dir, f"{commit_sha[:12]}.patch")
            with open(patch_path, "wb") as fh:  # write as binary
                fh.write(patch.stdout)
    except Exception as e:
        print(f"    [WARN] patch generation failed: {e}")
        
def after_dir_has_sources(after_dir: str) -> bool:
    """Return True if after/ already has .c/.h files."""
    if not os.path.exists(after_dir):
        return False
    existing = [f for f in os.listdir(after_dir)
                if f.endswith(".c") or f.endswith(".h")]
    return len(existing) > 0


# ===========================================================================
# MODE 1 — FIX COMMIT  (git diff based)
# ===========================================================================

def extract_fix_commit(test_dir: str, fix_sha: str, repo_name: str) -> bool:
    """
    Extract source files for the fixing commit.
    Discovers files via git diff (all C/H files changed in the fix commit).
    Returns True on success, False on failure.
    """
    commit_dir = os.path.join(test_dir, fix_sha)
    after_dir  = os.path.join(commit_dir, "after")
    before_dir = os.path.join(commit_dir, "before")
    fixing_dir = os.path.join(commit_dir, "fixing")

    # ── Skip if already done ──────────────────────────────────────────────
    if after_dir_has_sources(after_dir):
        existing = [f for f in os.listdir(after_dir)
                    if f.endswith(".c") or f.endswith(".h")]
        print(f"    [SKIP] already extracted ({len(existing)} files in after/)")
        return True

    repo_dir = os.path.join(REPOS_DIR, repo_name)

    if commit_has_no_parent(repo_dir, fix_sha):
        print(f"    [SKIP] orphan commit (no parent) — {fix_sha[:12]}")
        return False

    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir,  exist_ok=True)
    os.makedirs(fixing_dir, exist_ok=True)

    # ── Discover changed C/H files via git diff ───────────────────────────
    diff = subprocess.run(
        ["git", "diff", "--name-status", f"{fix_sha}^", fix_sha],
        cwd=repo_dir, capture_output=True, text=True, timeout=15
    )
    if diff.returncode != 0:
        print(f"    [FAIL] git diff failed: {diff.stderr[:200]}")
        return False

    files_to_extract = []
    for line in diff.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status   = parts[0]
        filepath = parts[-1]

        if status.startswith("D") or status.startswith("C"):
            continue
        if status.startswith("M"):
            if filepath.endswith(".c") or filepath.endswith(".h"):
                files_to_extract.append(
                    {"status": "M", "after": filepath, "before": filepath})
        elif status.startswith("A"):
            if filepath.endswith(".c") or filepath.endswith(".h"):
                files_to_extract.append(
                    {"status": "A", "after": filepath, "before": None})
        elif status.startswith("R") and len(parts) >= 3:
            old_path, new_path = parts[1], parts[2]
            if (old_path.endswith(".c") or old_path.endswith(".h") or
                    new_path.endswith(".c") or new_path.endswith(".h")):
                files_to_extract.append(
                    {"status": "R", "after": new_path, "before": old_path})

    if not files_to_extract:
        print(f"    [WARN] no C/H files changed in {fix_sha[:12]}")
        return True

    counts = {"M": 0, "A": 0, "R": 0}
    for f in files_to_extract:
        counts[f["status"]] += 1
    print(f"    Files: {len(files_to_extract)} "
          f"(M={counts['M']} A={counts['A']} R={counts['R']})")

    extracted = []
    for finfo in files_to_extract:
        underscore_name = finfo["after"].replace("/", "_")
        try:
            content = get_file_content(repo_name, fix_sha, finfo["after"])
            with open(os.path.join(after_dir, underscore_name), "w") as fh:
                fh.write(content)

            if finfo["before"] is None:
                with open(os.path.join(before_dir, underscore_name), "w") as fh:
                    fh.write("")
            else:
                content = get_file_content(
                    repo_name, f"{fix_sha}^", finfo["before"])
                with open(os.path.join(before_dir, underscore_name), "w") as fh:
                    fh.write(content)

            extracted.append(underscore_name)

        except subprocess.CalledProcessError as e:
            print(f"    [WARN] could not extract {finfo['after']}: {e}")
        except Exception as e:
            print(f"    [WARN] unexpected error for {finfo['after']}: {e}")

    write_commit_patch_file(fix_sha, fixing_dir, repo_dir)

    if extracted:
        print(f"    [OK] extracted {len(extracted)} files")
        return True
    else:
        print(f"    [FAIL] nothing extracted for {fix_sha[:12]}")
        return False


# ===========================================================================
# MODE 2 — HISTORY COMMITS  (fix-centric)
# ===========================================================================

def extract_history_commit(
    test_dir:      str,
    commit_sha:    str,
    repo_name:     str,
    fix_files_set: set,
) -> bool:
    """
    Extract source files for a single history commit using fix-centric
    approach — only extract files present in fix_files_set.
    Returns True on success, False on skip/failure.
    """
    commit_dir = os.path.join(test_dir, commit_sha)
    after_dir  = os.path.join(commit_dir, "after")
    before_dir = os.path.join(commit_dir, "before")
    fixing_dir = os.path.join(commit_dir, "fixing")

    # ── Skip if already done ──────────────────────────────────────────────
    if after_dir_has_sources(after_dir):
        existing = [f for f in os.listdir(after_dir)
                    if f.endswith(".c") or f.endswith(".h")]
        print(f"      [SKIP] already extracted ({len(existing)} files)")
        return True

    repo_dir = os.path.join(REPOS_DIR, repo_name)

    if commit_has_no_parent(repo_dir, commit_sha):
        print(f"      [SKIP] orphan commit (no parent)")
        return False

    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir,  exist_ok=True)
    os.makedirs(fixing_dir, exist_ok=True)

    extracted = []
    skipped   = []

    for underscore_name in fix_files_set:
        try:
            # Resolve actual git path in this commit
            git_path = resolve_underscore_name_to_git_path(repo_dir, commit_sha, underscore_name)
            if not git_path:
                git_path = find_unambiguous_file_by_basename(repo_dir, commit_sha, underscore_name)
                if git_path:
                    print(f"      [PATH-MOVE] {underscore_name} → {git_path}")
                else:
                    skipped.append(underscore_name)
                    continue

            # after/ — file at this commit
            try:
                content = get_file_content(repo_name, commit_sha, git_path)
                with open(os.path.join(after_dir, underscore_name), "w") as fh:
                    fh.write(content)
            except subprocess.CalledProcessError:
                skipped.append(underscore_name)
                continue

            # before/ — file at parent commit
            parent_path = resolve_underscore_name_to_git_path(
                repo_dir, f"{commit_sha}^", underscore_name)
            if not parent_path:
                parent_path = find_unambiguous_file_by_basename(
                    repo_dir, f"{commit_sha}^", underscore_name)

            if parent_path:
                try:
                    content = get_file_content(
                        repo_name, f"{commit_sha}^", parent_path)
                    with open(os.path.join(before_dir, underscore_name), "w") as fh:
                        fh.write(content)
                except subprocess.CalledProcessError:
                    with open(os.path.join(before_dir, underscore_name), "w") as fh:
                        fh.write("")
            else:
                with open(os.path.join(before_dir, underscore_name), "w") as fh:
                    fh.write("")

            extracted.append(underscore_name)

        except Exception as e:
            print(f"      [WARN] {underscore_name}: {e}")
            continue

    if not extracted:
        print(f"      [WARN] no files extracted — removing empty dirs")
        shutil.rmtree(commit_dir, ignore_errors=True)
        return False

    write_commit_patch_file(commit_sha, fixing_dir, repo_dir)

    print(f"      [OK] {len(extracted)} files  "
          f"(skipped {len(skipped)} not in commit)")
    return True


# ===========================================================================
# RUNNERS
# ===========================================================================

def run_fix(projects: list):
    """Mode 1 — extract fix commit source files."""
    total_tests = total_ok = total_fail = total_skip = 0

    for project in projects:
        repo_name   = PROJECT_TO_REPO[project]
        project_dir = os.path.join(DATA_DIR, project)

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
            test_dir  = os.path.join(project_dir, test_name)
            info_path = os.path.join(test_dir, "info.json")

            if not os.path.exists(info_path):
                print(f"\n  {test_name}: [SKIP] missing info.json")
                proj_skip += 1
                continue

            with open(info_path) as fh:
                info = json.load(fh)

            fix_sha = info.get("fix")
            if not fix_sha:
                print(f"\n  {test_name}: [SKIP] 'fix' key missing in info.json")
                proj_skip += 1
                continue

            print(f"\n  {test_name}  fix={fix_sha[:12]}")

            try:
                ok = extract_fix_commit(test_dir, fix_sha, repo_name)
                if ok:
                    proj_ok += 1
                else:
                    proj_fail += 1
            except Exception as e:
                print(f"    [ERROR] {e}")
                traceback.print_exc()
                proj_fail += 1

        total_tests += len(test_names)
        total_ok    += proj_ok
        total_fail  += proj_fail
        total_skip  += proj_skip

        print(f"\n  {project} summary — "
              f"ok={proj_ok}  fail={proj_fail}  skip={proj_skip}")

    print(f"\n{'='*70}")
    print(f"STEP 1 (FIX) COMPLETE")
    print(f"{'='*70}")
    print(f"  Total test cases : {total_tests}")
    print(f"  Extracted (ok)   : {total_ok}")
    print(f"  Failed           : {total_fail}")
    print(f"  Skipped          : {total_skip}")
    if total_tests > 0:
        print(f"  Success rate     : {total_ok*100/total_tests:.1f}%")


def run_history(projects: list):
    """Mode 2 — extract history commit source files (fix-centric)."""
    total_tests   = 0
    total_commits = ok_commits = skip_commits = fail_commits = 0

    for project in projects:
        repo_name   = PROJECT_TO_REPO[project]
        project_dir = os.path.join(DATA_DIR, project)

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

        for test_name in test_names:
            test_dir     = os.path.join(project_dir, test_name)
            info_path    = os.path.join(test_dir, "info.json")
            commits_path = os.path.join(test_dir, "commits.json")

            if not os.path.exists(info_path):
                continue
            if not os.path.exists(commits_path):
                print(f"\n  {test_name}: [SKIP] missing commits.json "
                      f"— run step 4b first")
                continue

            with open(info_path) as fh:
                info = json.load(fh)
            with open(commits_path) as fh:
                commits_data = json.load(fh)

            fix_sha         = info.get("fix", "")
            history_commits = commits_data.get("all_commits_in_history", [])

            if not history_commits:
                print(f"\n  {test_name}: [SKIP] no history commits")
                continue

            # Build fix_files_set from fix commit's after/
            fix_files_set = get_fix_files_from_after_dir(
                os.path.join(test_dir, fix_sha))
            if not fix_files_set:
                print(f"\n  {test_name}: [SKIP] fix commit after/ empty "
                      f"— run step 1 --mode fix first")
                continue

            print(f"\n  {test_name}  fix={fix_sha[:12]}  "
                  f"history={len(history_commits)}  "
                  f"fix_files={len(fix_files_set)}")

            total_tests   += 1
            test_ok = test_skip = test_fail = 0

            for commit_sha in history_commits:
                print(f"    {commit_sha[:12]}")

                try:
                    result = extract_history_commit(
                        test_dir, commit_sha, repo_name, fix_files_set)
                    if result:
                        test_ok += 1
                    else:
                        test_skip += 1
                except Exception as e:
                    print(f"      [ERROR] {e}")
                    traceback.print_exc()
                    test_fail += 1

            total_commits  += len(history_commits)
            ok_commits     += test_ok
            skip_commits   += test_skip
            fail_commits   += test_fail

            print(f"    → ok={test_ok}  skip={test_skip}  fail={test_fail}")

    print(f"\n{'='*70}")
    print(f"STEP 1 (HISTORY) COMPLETE")
    print(f"{'='*70}")
    print(f"  Test cases processed : {total_tests}")
    print(f"  Total commits        : {total_commits}")
    print(f"  Extracted (ok)       : {ok_commits}")
    print(f"  Skipped              : {skip_commits}")
    print(f"  Failed               : {fail_commits}")
    if total_commits > 0:
        print(f"  Success rate         : {ok_commits*100/total_commits:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract source files — fix commit (step 1) or history commits (step 5)"
    )
    parser.add_argument(
        "--mode", choices=["fix", "history"], default="fix",
        help="'fix' = fixing commit only (step 1), "
             "'history' = all V-SZZ history commits (step 5)"
    )
    parser.add_argument(
        "--projects", nargs="+",
        default=list(PROJECT_TO_REPO.keys()),
        choices=list(PROJECT_TO_REPO.keys()),
        help="Which projects to process (default: all configured projects)"
    )
    args = parser.parse_args()

    if args.mode == "fix":
        run_fix(projects=args.projects)
    else:
        run_history(projects=args.projects)