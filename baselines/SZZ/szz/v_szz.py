import sys
import logging as log
import traceback
from typing import List, Set

from git import Commit
from pydriller import ModificationType, GitRepository as PyDrillerGitRepo
import Levenshtein

from szz.core.abstract_szz import AbstractSZZ, ImpactedFile


def remove_whitespace(line_str):
    return ''.join(line_str.strip().split())


def compute_line_ratio(line_str1, line_str2):
    l1 = remove_whitespace(line_str1)
    l2 = remove_whitespace(line_str2)
    return Levenshtein.ratio(l1, l2)


MAXSIZE = sys.maxsize


class VSZZ(AbstractSZZ):
    """
    V-SZZ implementation.

    Traces deleted lines back through git history using Levenshtein
    similarity matching on diffs to follow line evolution.
    """

    def __init__(self, repo_full_name: str, repo_url: str, repos_dir: str = None, use_temp_dir: bool = True):
        super().__init__(repo_full_name, repo_url, repos_dir, use_temp_dir)

    def find_bic(self, fix_commit_hash: str, impacted_files: List['ImpactedFile'], **kwargs) -> Set[Commit]:
        """
        Find bug introducing commits candidates by tracing line history.

        :param str fix_commit_hash: hash of fix commit to scan for buggy commits
        :param List[ImpactedFile] impacted_files: list of impacted files in fix commit
        :key ignore_revs_file_path (str): specify ignore revs file for git blame to ignore specific commits.
        :returns list of line history records with previous_commits chains
        """

        log.info(f"find_bic() kwargs: {kwargs}")

        ignore_revs_file_path = kwargs.get('ignore_revs_file_path', None)

        bug_introd_commits = []
        for imp_file in impacted_files:
            try:
                blame_data = self._blame(
                    rev='{commit_id}^'.format(commit_id=fix_commit_hash),
                    file_path=imp_file.file_path,
                    modified_lines=imp_file.modified_lines,
                    ignore_revs_file_path=ignore_revs_file_path,
                    ignore_whitespaces=True,
                    skip_comments=True
                )

                for entry in blame_data:
                    previous_commits = []

                    blame_result = entry
                    while True:
                        mapped_line_num = self.map_modified_line(blame_result, imp_file.file_path)

                        previous_commits.append({
                            "commit": blame_result.commit.hexsha,
                            "line number": blame_result.line_num,
                            "line content": blame_result.line_str,
                            "hunk(diff message)": blame_result.hunk,
                        })

                        if mapped_line_num == -1:
                            break

                        blame_data2 = self._blame(
                            rev='{commit_id}^'.format(commit_id=blame_result.commit.hexsha),
                            file_path=imp_file.file_path,
                            modified_lines=[mapped_line_num],
                            ignore_revs_file_path=ignore_revs_file_path,
                            ignore_whitespaces=True,
                            skip_comments=True
                        )
                        blame_result = list(blame_data2)[0]

                    bug_introd_commits.append({
                        'line_num': entry.line_num,
                        'line_str': entry.line_str,
                        'file_path': entry.file_path,
                        'content around': entry.context_around_line,
                        'previous_commits': previous_commits,
                    })
            except:
                print(traceback.format_exc())

        return bug_introd_commits

    def map_modified_line(self, blame_entry, blame_file_path):
        """
        Map a blamed line to its predecessor in the parent commit's diff
        using Levenshtein similarity matching.

        Returns the source line number in the parent, or -1 if the line
        was newly added (no predecessor).
        """
        blame_commit = PyDrillerGitRepo(self.repository_path).get_commit(blame_entry.commit.hexsha)

        for mod in blame_commit.modifications:
            file_path = mod.new_path
            if mod.change_type == ModificationType.DELETE or mod.change_type == ModificationType.RENAME:
                file_path = mod.old_path

            if file_path != blame_file_path:
                continue

            if not mod.old_path:
                # newly added file
                return -1

            lines_deleted = [deleted for deleted in mod.diff_parsed['deleted']]

            if len(lines_deleted) == 0:
                return -1

            if blame_entry.line_str:
                sorted_lines_deleted = [
                    (line[0], line[1],
                     compute_line_ratio(blame_entry.line_str, line[1]),
                     abs(blame_entry.line_num - line[0]))
                    for line in lines_deleted
                ]
                sorted_lines_deleted = sorted(
                    sorted_lines_deleted,
                    key=lambda x: (x[2], MAXSIZE - x[3]),
                    reverse=True
                )

                if sorted_lines_deleted[0][2] > 0.75:
                    return sorted_lines_deleted[0][0]

        return -1