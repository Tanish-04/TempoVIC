import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from setting import *
sys.path.append(SZZ_FOLDER)

from szz.ag_szz import AGSZZ
from szz.b_szz import BaseSZZ
from szz.ma_szz import MASZZ, DetectLineMoved
from szz.v_szz import VSZZ
from data_loader import load_annotated_commits, load_project


@dataclass
class SZZConfig:
    """Configuration class for SZZ parameters"""
    method: str
    language: str
    time: str
    max_change_size: int = DEFAULT_MAX_CHANGE_SIZE


class DualOutput:
    """Class to handle dual output to both terminal and file"""

    def __init__(self, filename: str, mode: str = 'w'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.file = open(filename, mode)
        self.terminal_stdout = sys.stdout

    def write(self, message: str) -> None:
        self.terminal_stdout.write(message)
        self.file.write(message)

    def flush(self) -> None:
        if hasattr(self.terminal_stdout, 'flush'):
            self.terminal_stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class SZZRunner:
    """Main class to handle SZZ execution"""

    def __init__(self, config: SZZConfig):
        self.config = config
        self.use_temp_dir = False

    def _convert_project_name(self, project: str) -> str:
        """Convert project name by replacing '/' with '_'"""
        return project.replace("/", "_") if "/" in project else project

    def _get_output_paths(self, project: str) -> tuple:
        """Get output file and progress file paths"""
        pro_name = self._convert_project_name(project)
        output_file = f"results/{self.config.method}-szz/{self.config.language}/{self.config.time}/{self.config.method}-{pro_name}.json"
        progress_file = f"results/{self.config.method}-szz/{self.config.language}/{self.config.time}/{self.config.method}-{pro_name}-progress.json"
        return output_file, progress_file

    def _load_existing_output(self, output_file: str) -> Dict[str, Any]:
        """Load existing output from file"""
        if os.path.exists(output_file):
            with open(output_file, 'r') as fin:
                return json.load(fin)
        return {}

    def _load_completed_commits(self, progress_file: str) -> List[str]:
        """Load list of completed commits from progress file"""
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as fin:
                return json.load(fin)
        return []

    def _save_output(self, output: Dict[str, Any], output_file: str) -> None:
        """Save output to file"""
        with open(output_file, 'w') as fout:
            json.dump(output, fout, indent=4)

    def _save_progress(self, completed_commits: List[str], progress_file: str) -> None:
        """Save progress to file"""
        with open(progress_file, 'w') as fout:
            json.dump(completed_commits, fout, indent=4)

    def _run_base_szz(self, project: str, commits: List[str], repo_url: Optional[str] = None) -> Dict[str, Any]:
        """Run Base SZZ method"""
        output = {}
        b_szz = BaseSZZ(repo_full_name=project, repo_url=repo_url, repos_dir=REPOS_DIR, use_temp_dir=self.use_temp_dir)

        for commit in commits:
            print("==================================================")
            print('Fixing Commit:', commit)

            imp_files = b_szz.get_impacted_files(
                fix_commit_hash=commit,
                file_ext_to_parse=['c', 'h'],
                only_deleted_lines=True
            )
            bug_inducing_commits = b_szz.find_bic(
                fix_commit_hash=commit,
                impacted_files=imp_files,
                ignore_revs_file_path=None
            )
            output[commit] = [c.hexsha for c in bug_inducing_commits]

        return output

    def _run_ag_szz(self, project: str, commits: List[str], repo_url: Optional[str] = None) -> Dict[str, Any]:
        """Run AG SZZ method"""
        output = {}
        ag_szz = AGSZZ(repo_full_name=project, repo_url=repo_url, repos_dir=REPOS_DIR, use_temp_dir=self.use_temp_dir)

        for commit in commits:
            print('Fixing Commit:', commit)

            imp_files = ag_szz.get_impacted_files(
                fix_commit_hash=commit,
                file_ext_to_parse=['c', 'h'],
                only_deleted_lines=True
            )
            bug_inducing_commits = ag_szz.find_bic(
                fix_commit_hash=commit,
                impacted_files=imp_files,
                ignore_revs_file_path=None,
                max_change_size=self.config.max_change_size
            )
            output[commit] = [c.hexsha for c in bug_inducing_commits]

        return output

    def _run_ma_szz(self, project: str, commits: List[str], repo_url: Optional[str] = None) -> Dict[str, Any]:
        """Run MA SZZ method"""
        output = {}
        ma_szz = MASZZ(repo_full_name=project, repo_url=repo_url, repos_dir=REPOS_DIR, use_temp_dir=self.use_temp_dir)

        for commit in commits:
            print('Fixing Commit:', commit)

            imp_files = ma_szz.get_impacted_files(
                fix_commit_hash=commit,
                file_ext_to_parse=['c', 'h'],
                only_deleted_lines=True
            )
            bug_inducing_commits = ma_szz.find_bic(
                fix_commit_hash=commit,
                impacted_files=imp_files,
                ignore_revs_file_path=None,
                max_change_size=self.config.max_change_size
            )
            output[commit] = [c.hexsha for c in bug_inducing_commits]

        return output

    def _run_v_szz(self, project: str, commits: List[str], repo_url: Optional[str] = None) -> Dict[str, Any]:
        """Run V SZZ method"""
        output = {}
        v_szz = VSZZ(repo_full_name=project, repo_url=repo_url, repos_dir=REPOS_DIR, use_temp_dir=self.use_temp_dir)

        for commit in commits:
            print('Fixing Commit:', commit)

            imp_files = v_szz.get_impacted_files(
                fix_commit_hash=commit,
                file_ext_to_parse=['c', 'h'],
                only_deleted_lines=True
            )
            bug_inducing_commits = v_szz.find_bic(
                fix_commit_hash=commit,
                impacted_files=imp_files,
                ignore_revs_file_path=None
            )
            output[commit] = bug_inducing_commits

        return output

    def run_szz(self, project: str, commits: List[str], repo_url: Optional[str] = None, repo_name: Optional[str] = None) -> None:
        """Main method to run SZZ based on configuration"""
        method_handlers = {
            "b": self._run_base_szz,
            "ag": self._run_ag_szz,
            "ma": self._run_ma_szz,
            "v": self._run_v_szz,
        }

        if self.config.method not in method_handlers:
            raise ValueError(f"Unsupported method: {self.config.method}. "
                             f"Choose from: {list(method_handlers.keys())}")

        repo_to_use = repo_name if repo_name else project
        output = method_handlers[self.config.method](repo_to_use, commits, repo_url)

        output_file, _ = self._get_output_paths(project)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self._save_output(output, output_file)


class Logger:
    """Logger class to handle output redirection"""

    @staticmethod
    def setup_logging(config: SZZConfig) -> None:
        """Setup logging with dual output"""
        base_filename = f'results/log/{config.method}-szz__{config.language}__generate-re__log__{config.time}.txt'
        mode = 'a' if os.path.exists(base_filename) else 'w'
        sys.stdout = sys.stderr = DualOutput(base_filename, mode)


def print_config(config: SZZConfig) -> None:
    """Print configuration parameters"""
    print(f"Method: {config.method}")
    print(f"Language: {config.language}")
    print(f"Time: {config.time}")


def parse_arguments() -> SZZConfig:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SZZ Implementation Runner')

    parser.add_argument('--method', type=str, default='b',
                        choices=['b', 'ag', 'ma', 'v'],
                        help='SZZ variant to run (default: b)')
    parser.add_argument('--language', type=str, default='C',
                        help='Programming language (default: C)')
    parser.add_argument('--time', type=str, default='run1',
                        help='Run identifier for output directory (default: run1)')

    args = parser.parse_args()

    return SZZConfig(
        method=args.method,
        language=args.language,
        time=args.time,
    )


def main() -> None:
    """Main function"""
    config = parse_arguments()
    print_config(config)

    # Setup logging
    Logger.setup_logging(config)

    # Initialize SZZ runner
    runner = SZZRunner(config)

    # Load projects and commits
    projects = load_project(config.language)
    project_commits = load_annotated_commits()

    # Process only projects that exist in both the dataset and locally
    project_list = sorted(set(projects).intersection(set(project_commits.keys())))

    # Mapping for common differences between dataset project names and repository folders
    project_to_repo = {
        "linux-kernel": "linux",
        "linux": "linux",
        "FFmpeg": "FFmpeg",
        "ImageMagick": "ImageMagick",
        "OpenSSL": "OpenSSL",
        "PHP-SRC": "PHP-SRC",
    }

    for project in project_list:
        repo_name = project_to_repo.get(project, project)
        repo_folder = os.path.join(REPOS_DIR, repo_name)

        if not os.path.exists(repo_folder):
            print(f"Skipping {project} as repository does not exist at {repo_folder}")
            continue

        print(f"\nProject: {project} (Repo: {repo_name})")
        print(f"  Commits to process: {len(project_commits[project])}")

        runner.run_szz(project, project_commits[project], repo_name=repo_name)


if __name__ == "__main__":
    main()