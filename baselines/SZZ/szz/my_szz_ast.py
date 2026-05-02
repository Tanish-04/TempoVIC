import os
import sys
import logging as log
import traceback
from typing import List, Set, Dict, Any
import subprocess
import json

from git import Commit

from szz.core.abstract_szz import AbstractSZZ, ImpactedFile

from pydriller import ModificationType, GitRepository as PyDrillerGitRepo
import Levenshtein

MAXSIZE = sys.maxsize


def _remove_whitespace(line_str: str) -> str:
    """Remove all whitespace for fuzzy comparison."""
    return ''.join(line_str.strip().split())


def _compute_line_ratio(line_str1: str, line_str2: str) -> float:
    """Levenshtein similarity between two lines (whitespace-normalized)."""
    l1 = _remove_whitespace(line_str1)
    l2 = _remove_whitespace(line_str2)
    if not l1 and not l2:
        return 1.0
    if not l1 or not l2:
        return 0.0
    return Levenshtein.ratio(l1, l2)


class MySZZAST(AbstractSZZ):
    """
    V-SZZ implementation that traces git blame iteratively with
    diff-based line mapping to find the EARLIEST vulnerability
    inducing commit.

    This matches the algorithm in my_szz.py (Levenshtein similarity
    on deleted lines) — the only authoritative approach for C files.

    Supported **kwargs:
    * ignore_revs_file_path: Path to file with commits to ignore in git blame
    """

    def __init__(self, repo_full_name: str, repo_url: str, repos_dir: str = None,
                 use_temp_dir: bool = True, ast_map_path=None):
        super().__init__(repo_full_name, repo_url, repos_dir, use_temp_dir)
        self.ast_map_path = ast_map_path

        print(f"[V-SZZ] Initialized")
        print(f"[V-SZZ] Repository: {self.repository_path}")

    # map_modified_line — the CORE of V-SZZ line tracking
    def map_modified_line(self, blame_entry, blame_file_path: str):
        """
        Given a blame result (commit + line), map the line back to the
        PARENT commit using the commit's diff.

        Returns a tuple: (old_line_num, old_file_path)
          - (N, path)  → the line existed at line N in path in the parent
          - (-1, path)  → line was INSERTED by this commit (introducer)

        Handles two tricky cases the original missed:
        1. Line was MODIFIED (not just moved): the tracked line appears in
           ADDED but the old version in DELETED is too different. We use
           line-proximity matching to find the corresponding deleted line.
        2. File was RENAMED: blame_file_path is the path at the fix commit,
           but in older commits the file lived at a different path.

        :param blame_entry: BlameData with .commit, .line_num, .line_str
        :param blame_file_path: path of the file being tracked
        :returns tuple: (old_line_num, old_file_path)
                        old_line_num=-1 means this commit is the introducer
        """
        blame_commit = PyDrillerGitRepo(self.repository_path).get_commit(
            blame_entry.commit.hexsha
        )

        # --- Step 1: Find the modification that touches our file ---
        # Must handle file renames: blame_file_path may match mod.new_path
        # OR mod.old_path (Linux kernel files move between directories).
        target_mod = None
        for mod in blame_commit.modifications:
            paths = set()
            if mod.new_path:
                paths.add(mod.new_path)
            if mod.old_path:
                paths.add(mod.old_path)
            if blame_file_path in paths:
                target_mod = mod
                break

        if target_mod is None:
            # No exact path match — try suffix matching for renamed files.
            # e.g., "drivers/media/usb/gspca/foo.c" should match
            #        "drivers/media/video/gspca/foo.c"
            basename = os.path.basename(blame_file_path)
            candidates = []
            for mod in blame_commit.modifications:
                for p in (mod.new_path, mod.old_path):
                    if p and os.path.basename(p) == basename:
                        candidates.append(mod)
                        break

            if len(candidates) == 1:
                target_mod = candidates[0]
            elif len(candidates) > 1:
                # Multiple files with same basename — pick best content match
                best_ratio = 0
                for mod in candidates:
                    for _, lc in mod.diff_parsed.get('added', []):
                        ratio = _compute_line_ratio(blame_entry.line_str, lc)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            target_mod = mod

        if target_mod is None:
            # Still no match — treat as introducer
            return (-1, blame_file_path)

        mod = target_mod

        # Determine the old file path (for tracking through renames)
        old_file_path = mod.old_path if mod.old_path else blame_file_path

        if not mod.old_path:
            # File was newly added → this commit introduced it
            return (-1, blame_file_path)

        lines_deleted = mod.diff_parsed['deleted']
        lines_added = mod.diff_parsed['added']

        # --- Step 2: Try direct content match in DELETED lines ---
        if blame_entry.line_str and lines_deleted:
            scored = [
                (
                    line[0],                                        # old line number
                    line[1],                                        # old line content
                    _compute_line_ratio(blame_entry.line_str, line[1]),  # similarity
                    abs(blame_entry.line_num - line[0])             # distance tie-break
                )
                for line in lines_deleted
            ]
            scored.sort(key=lambda x: (x[2], MAXSIZE - x[3]), reverse=True)

            if scored[0][2] > 0.75:
                return (scored[0][0], old_file_path)


        # --- Step 4: No deleted lines at all ---
        if len(lines_deleted) == 0:
            if lines_added:
                # Pure addition, no deletions → line was introduced here
                return (-1, blame_file_path)
            # No diff at all for this file (shouldn't happen)
            return (-1, blame_file_path)

        # No match found → this commit introduced the line
        return (-1, blame_file_path)

    # ------------------------------------------------------------------
    # find_bic — main entry point
    # ------------------------------------------------------------------
    def find_bic(self, fix_commit_hash: str, impacted_files: List['ImpactedFile'],
                 **kwargs) -> List[Dict[str, Any]]:
        """
        Find the EARLIEST bug introducing commit for each deleted line.

        Algorithm:
          For each deleted line in the fix commit:
          1. git blame fix_commit^ → commit_A
          2. map_modified_line(commit_A) → old_line_num in commit_A^
             - If -1 → commit_A is the introducer, stop
          3. git blame commit_A^ at old_line_num → commit_B
          4. Repeat from step 2 with commit_B

        :param str fix_commit_hash: hash of fix commit
        :param List[ImpactedFile] impacted_files: impacted files with deleted lines
        :key ignore_revs_file_path (str): commits to ignore in git blame
        :returns List[Dict] line histories with introducer commits
        """
        log.info(f"[V-SZZ] find_bic() kwargs: {kwargs}")
        ignore_revs_file_path = kwargs.get('ignore_revs_file_path', None)

        print(f"[V-SZZ] Finding EARLIEST vulnerability inducing commits")
        print(f"[V-SZZ] Fix commit: {fix_commit_hash[:12]}")

        bug_introd_commits = []

        for imp_file in impacted_files:
            print(f"\n[V-SZZ] Processing file: {imp_file.file_path}")
            print(f"[V-SZZ] Modified lines: {imp_file.modified_lines}")

            try:
                # Step 1: Initial git blame on fix_commit^
                blame_data = self._blame(
                    rev='{commit_id}^'.format(commit_id=fix_commit_hash),
                    file_path=imp_file.file_path,
                    modified_lines=imp_file.modified_lines,
                    ignore_revs_file_path=ignore_revs_file_path,
                    ignore_whitespaces=False,
                    skip_comments=True
                )

                for entry in blame_data:
                    print(f"\n[V-SZZ] Tracking line {entry.line_num}: "
                          f"{entry.line_str[:60]}...")

                    # Iteratively trace back using map_modified_line
                    previous_commits = []
                    blame_result = entry
                    # Track the current file path — may change on renames
                    current_file_path = imp_file.file_path

                    max_iterations = 500  # Safety limit
                    iteration = 0

                    while iteration < max_iterations:
                        iteration += 1

                        # Step 2: Map line through this commit's diff
                        # Returns (old_line_num, old_file_path) tuple
                        mapped_result = self.map_modified_line(
                            blame_result, current_file_path
                        )
                        mapped_line_num, mapped_file_path = mapped_result

                        # Record this commit in the history chain
                        previous_commits.append((
                            blame_result.commit.hexsha,
                            blame_result.line_num,
                            blame_result.line_str,
                            current_file_path
                        ))

                        print(f"[V-SZZ]   [{iteration}] {blame_result.commit.hexsha[:12]} "
                              f"line {blame_result.line_num} → "
                              f"mapped_old_line={mapped_line_num}"
                              f"{' (rename: ' + mapped_file_path + ')' if mapped_file_path != current_file_path else ''}")

                        if mapped_line_num == -1:
                            # Line was INSERTED by this commit → it's the introducer
                            print(f"[V-SZZ]   ✓ Introducer: "
                                  f"{blame_result.commit.hexsha[:12]}")
                            break

                        # Update file path if a rename was detected
                        if mapped_file_path != current_file_path:
                            print(f"[V-SZZ]   ⤷ File renamed: "
                                  f"{current_file_path} → {mapped_file_path}")
                            current_file_path = mapped_file_path

                        # Step 3: Blame parent at the MAPPED old line number
                        try:
                            blame_data2 = self._blame(
                                rev='{commit_id}^'.format(
                                    commit_id=blame_result.commit.hexsha),
                                file_path=current_file_path,
                                modified_lines=[mapped_line_num],
                                ignore_revs_file_path=ignore_revs_file_path,
                                ignore_whitespaces=False,
                                skip_comments=True
                            )
                            blame_entries = list(blame_data2)

                            if not blame_entries:
                                print(f"[V-SZZ]   ✓ Blame returned empty — "
                                      f"introducer: {blame_result.commit.hexsha[:12]}")
                                break

                            blame_result = blame_entries[0]

                        except Exception:
                            # Blame failed → parent doesn't have this line
                            print(f"[V-SZZ]   ✓ Blame failed at parent — "
                                  f"introducer: {blame_result.commit.hexsha[:12]}")
                            break

                    if iteration >= max_iterations:
                        print(f"[V-SZZ]   ⚠ Reached iteration limit ({max_iterations})")

                    # The introducer is the LAST commit in previous_commits
                    introducer = (previous_commits[-1][0] if previous_commits
                                  else entry.commit.hexsha)
                    print(f"[V-SZZ]   → EARLIEST introducer: {introducer[:12]} "
                          f"(traced {len(previous_commits)} commits)")

                    bug_introd_commits.append({
                        'line_num': entry.line_num,
                        'line_str': entry.line_str,
                        'file_path': imp_file.file_path,
                        'previous_commits': previous_commits
                    })

            except Exception as e:
                print(f"[V-SZZ] Error processing file {imp_file.file_path}: {e}")
                print(traceback.format_exc())

        print(f"\n[V-SZZ] Found {len(bug_introd_commits)} line histories")
        return bug_introd_commits
