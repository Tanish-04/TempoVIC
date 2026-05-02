import os
import subprocess
import git
from unidiff import PatchSet
from io import StringIO


def is_noise(line):
    """Check if a line is noise (blank or C-style comment)."""
    line = line.strip('\t').strip('\r').strip()
    if line == '':
        return True
    if line.startswith('//') or line.startswith('/**') or line.startswith('*') or \
            line.startswith('/*') or line.endswith('*/'):
        return True
    return False


class GitLog:
    commands = {
        'meta': 'meta_cmd',
        'numstat': 'numstat_cmd',
        'namestat': 'namestat_cmd',
        'merge_numstat': 'merge_numstat_cmd',
        'merge_namestat': 'merge_namestat_cmd'
    }

    def __init__(self):
        self.meta_cmd = 'git log --reverse --all --pretty=format:\"commit: %H%n' \
                        'parent: %P%n' \
                        'author: %an%n' \
                        'author email: %ae%n' \
                        'time stamp: %at%n' \
                        'committer: %cn%n' \
                        'committer email: %ce%n' \
                        '%B%n\"  '
        self.numstat_cmd = 'git log --pretty=format:\"commit: %H\" --numstat -M --all --reverse '
        self.namestat_cmd = 'git log  --pretty=format:\"commit: %H\" --name-status -M --all --reverse '
        self.merge_numstat_cmd = 'git log --pretty=oneline --numstat -m --merges -M --all --reverse '
        self.merge_namestat_cmd = 'git log --pretty=oneline  --name-status -m --merges -M  --all --reverse '

    def git_log(self, project_path):
        cmd = getattr(self, GitLog.commands.get('meta'))
        out = subprocess.check_output(
            cmd, shell=True, cwd=project_path
        ).decode('utf-8', errors='ignore')
        return out

    def git_tag(self, project_path):
        out = subprocess.check_output(
            'git tag', shell=True, cwd=project_path
        ).decode('utf-8', errors='ignore')
        return out

    def git_show(self, project_path, tag):
        cmd = f'git show {tag} --pretty=format:"commit: %H%ntimestamp: %ct%n"'
        out = subprocess.check_output(
            cmd, shell=True, cwd=project_path
        ).decode('utf-8', errors='ignore')
        return out

    def git_diff(self, project_path, commit_id):
        repository = git.Repo(project_path)
        try:
            uni_diff_text = repository.git.diff(
                commit_id + '~1', commit_id,
                ignore_blank_lines=True,
                ignore_space_at_eol=True
            )
            patch_set = PatchSet(StringIO(uni_diff_text))
        except Exception as e:
            print(project_path, 'Error: ', e)
            return None

        change_list = []
        for patched_file in patch_set:
            file_path = patched_file.path
            ad_line_no = [
                (line.target_line_no, line.value)
                for hunk in patched_file for line in hunk
                if line.is_added and not is_noise(line.value.strip())
            ]
            del_line_no = [
                (line.source_line_no, line.value)
                for hunk in patched_file for line in hunk
                if line.is_removed and not is_noise(line.value.strip())
            ]
            change_list.append((file_path, del_line_no, ad_line_no))

        return change_list

    def git_diff_2(self, project_path, commit_id):
        repository = git.Repo(project_path)
        try:
            uni_diff_text = repository.git.diff(
                commit_id + '~1', commit_id,
                ignore_blank_lines=True,
                ignore_space_at_eol=True
            )
            patch_set = PatchSet(StringIO(uni_diff_text))
        except Exception as e:
            print(project_path, 'Error: ', e)
            return None

        change_list = []
        for patched_file in patch_set:
            file_path = patched_file.path
            ad_line_no = [
                (line.target_line_no, line.value)
                for hunk in patched_file for line in hunk
                if line.is_added and not is_noise(line.value.strip())
            ]
            del_line_no = [
                (line.source_line_no, line.value)
                for hunk in patched_file for line in hunk
                if line.is_removed and not is_noise(line.value.strip())
            ]
            change_list.append({
                "file_path": file_path,
                "del_line_no": del_line_no,
                "ad_line_no": ad_line_no,
                "is_added_file": patched_file.is_added_file,
            })

        return change_list

    def get_commit_time(self, project_path, commit_id):
        cmd = f'git show -s --format=%ci {commit_id}'
        out = subprocess.check_output(
            cmd, shell=True, cwd=project_path
        ).decode('utf-8', errors='ignore')
        return out

    def fetch_tags(self, project_path):
        subprocess.check_output(
            'git fetch --all --tags', shell=True, cwd=project_path
        ).decode('utf-8', errors='ignore')

    def get_tags(self, project_path):
        out = subprocess.check_output(
            'git show-ref --tags', shell=True, cwd=project_path
        ).decode('utf-8', errors='ignore')
        return out

    def get_commits_range(self, project_path, commit1, commit2):
        cmd = f'git log --pretty=oneline {commit1}...{commit2}'
        out = subprocess.check_output(
            cmd, shell=True, cwd=project_path
        ).decode('utf-8', errors='ignore')
        return out

    def get_commits_from(self, project_path, commit_id):
        cmd = f'git log --pretty=oneline {commit_id}'
        out = subprocess.check_output(
            cmd, shell=True, cwd=project_path
        ).decode('utf-8', errors='ignore')
        commits = [line.split(' ')[0] for line in out.split('\n') if line]
        return commits
