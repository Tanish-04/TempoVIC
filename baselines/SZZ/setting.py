"""
Shared constants for the SZZ baselines module.

All paths are relative to this file's parent directory (baselines/SZZ/).
Adjust REPOS_DIR and DATA_FOLDER to point at your local data.
"""
import os
from pathlib import Path

# Root of this module
HERE = Path(__file__).resolve().parent

# Root of the TempoVIC project (two levels up from baselines/SZZ/)
PROJECT_ROOT = HERE.parent.parent

# Directory containing cloned git repositories (one folder per project).
# Same repos used by graph_construction.
REPOS_DIR = str(PROJECT_ROOT / "graph_construction" / "repositories")

# Directory containing the CVE dataset JSON files
# (verified_cve_with_versions_C.json, etc.)
DATA_FOLDER = str(HERE / "data")

# Working directory for SZZ output (results/, logs, etc.)
WORK_DIR = str(HERE)

# Path where SZZ implementation modules live
SZZ_FOLDER = str(HERE)

# Default maximum change size threshold for AG-SZZ and MA-SZZ
DEFAULT_MAX_CHANGE_SIZE = 20
