"""
Shared configuration for the NeuralSZZ baseline.
All paths are resolved relative to this file's location.
"""
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# Graph construction data (test cases with info.json, graph.json, graph_bszz.json)
GRAPH_DATA_DIR = PROJECT_ROOT / "graph_construction" / "data"

# Git repositories
REPOS_DIR = str(PROJECT_ROOT / "graph_construction" / "repositories")

# SZZ implementation path (for B-SZZ imports)
SZZ_DIR = str(PROJECT_ROOT / "baselines" / "SZZ")

# Local data directory for NeuralSZZ intermediate files (miniGraphs, etc.)
NEURALSZZ_DATA_DIR = SCRIPT_DIR / "data"

# Project name → repository folder mapping
PROJECT_TO_REPO = {
    "linux": "linux",
    "FFmpeg": "FFmpeg",
    "ImageMagick": "ImageMagick",
    "OpenSSL": "OpenSSL",
    "PHP-SRC": "PHP-SRC",
}


def get_available_projects():
    """Return list of projects that have data directories."""
    return sorted(
        name for name in PROJECT_TO_REPO
        if (GRAPH_DATA_DIR / name).is_dir()
    )
