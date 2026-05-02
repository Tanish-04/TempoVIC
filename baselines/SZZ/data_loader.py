"""
Data loader for the SZZ baselines.

Loads CVE datasets from JSON files in DATA_FOLDER and provides
project-to-commit mappings used by main.py and evaluate.py.
"""

import os
import json

from setting import DATA_FOLDER


def load_project(language):
    """Load the unique set of project names for a given language."""
    file_path = os.path.join(DATA_FOLDER, f'verified_cve_with_versions_{language}.json')

    with open(file_path) as fin:
        cve_data = json.load(fin)

    projects = set()
    for entry in cve_data:
        project_name = entry.get("project")
        if project_name:
            projects.add(project_name)

    return sorted(list(projects))


def load_annotated_commits(target_projects=None):
    """
    Load all annotated CVE commits, returning a dict mapping
    project name -> list of fixing commit hashes.
    """
    all_files = [
        f for f in os.listdir(DATA_FOLDER)
        if f.startswith('verified_cve_with_versions_') and f.endswith('.json')
    ]

    merged_data = []

    for file_name in all_files:
        file_path = os.path.join(DATA_FOLDER, file_name)

        if os.path.getsize(file_path) == 0:
            print(f"Warning: The file {file_name} is empty, skipping.")
            continue

        try:
            with open(file_path) as fin:
                cve_data = json.load(fin)
                merged_data.extend(cve_data)
        except json.JSONDecodeError as e:
            print(f"Error loading JSON from file {file_name}: {e}")
            continue

    project_commits = {}

    for item in merged_data:
        project_name = item['project']
        fixing_commits = [fd['fixing_commit'] for fd in item['fixing_details']]

        if project_name in project_commits:
            project_commits[project_name].extend(fixing_commits)
        else:
            project_commits[project_name] = fixing_commits

    return project_commits


def fixing_commit_to_cve():
    """
    Build a mapping from fixing commit hash -> CVE ID.
    """
    all_files = [
        f for f in os.listdir(DATA_FOLDER)
        if f.startswith('verified_cve_with_versions_') and f.endswith('.json')
    ]

    merged_data = []

    for file_name in all_files:
        file_path = os.path.join(DATA_FOLDER, file_name)
        if os.path.getsize(file_path) == 0:
            continue

        try:
            with open(file_path) as fin:
                cve_data = json.load(fin)
                merged_data.extend(cve_data)
        except json.JSONDecodeError:
            continue

    mapping = {}
    for item in merged_data:
        for fixing in item['fixing_details']:
            mapping[fixing['fixing_commit']] = item["cve_id"]

    return mapping
