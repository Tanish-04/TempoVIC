import os
import subprocess

from config_loader import (
    get_repos_dir,
    get_data_dir,
    get_all_projects,
    get_project_repo_name,
)

REPOS_DIR = get_repos_dir()
DATA_DIR  = get_data_dir()

PROJECT_TO_REPO = {
    project: get_project_repo_name(project)
    for project in get_all_projects()
}


def get_file_content(repo_name: str, commit: str, filepath: str) -> str:
    """Return the content of a file at a specific commit."""
    repo_dir = os.path.join(REPOS_DIR, repo_name)
    return subprocess.check_output(
        ["git", "show", f"{commit}:{filepath}"],
        cwd=repo_dir,
    ).decode("utf-8", errors="ignore")


def resolve_underscore_name_to_git_path(repo_dir: str, commit: str, underscore_name: str, retry: int = 2) -> str:
    """
    Convert an underscore-format filename (e.g. drivers_tty_n_tty.c) back to
    its real git path by listing all files in that commit.
    Returns None if not found.
    """
    import time
    for attempt in range(retry + 1):
        try:
            result = subprocess.run(
                ["git", "ls-tree", "-r", "--name-only", commit],
                cwd=repo_dir,
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                if attempt < retry:
                    time.sleep(0.5)
                    continue
                return None

            for git_path in result.stdout.strip().split('\n'):
                if git_path.endswith('.c') or git_path.endswith('.h'):
                    if git_path.replace('/', '_') == underscore_name:
                        return git_path
            return None

        except subprocess.TimeoutExpired:
            if attempt < retry:
                time.sleep(1)
                continue
            return None
        except Exception:
            if attempt < retry:
                time.sleep(0.5)
                continue
            return None

    return None


def find_unambiguous_file_by_basename(repo_dir: str, commit: str, underscore_name: str) -> str:
    """
    match by base filename only (handles file moves between commits).
    Returns a match only when it is unambiguous (exactly one result).
    """
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit],
            cwd=repo_dir,
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None

        c_files = [f for f in result.stdout.strip().split('\n')
                   if f.endswith('.c') or f.endswith('.h')]
        parts = underscore_name.split('_')

        for i in range(min(3, len(parts)), 1, -1):
            matches = [f for f in c_files
                       if os.path.basename(f) == '_'.join(parts[-i:])]
            if len(matches) == 1:
                return matches[0]

        return None
    except Exception:
        return None


def commit_has_no_parent(repo_dir: str, commit_sha: str) -> bool:
    """Return True if the commit has no parent."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{commit_sha}^"],
        cwd=repo_dir, capture_output=True, text=True, timeout=5
    )
    return result.returncode != 0


def get_fix_files_from_after_dir(commit_dir: str) -> set:
    """
    Return the set of .c/.h filenames (underscore format) from a commit's
    after/ directory. Used by later steps to apply fix-centric filtering.
    Returns an empty set if the directory does not exist or is empty.
    """
    after_dir = os.path.join(commit_dir, "after")
    if not os.path.exists(after_dir):
        return set()
    try:
        return {f for f in os.listdir(after_dir)
                if f.endswith('.c') or f.endswith('.h')}
    except Exception:
        return set()