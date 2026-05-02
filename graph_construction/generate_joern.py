"""
Usage:
    python generate_joern.py --projects linux
    python generate_joern.py --projects FFmpeg OpenSSL PHP-SRC ImageMagick
    python generate_joern.py --projects all
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import (
    get_all_projects,
    get_joern_script,
    get_joern_targets_file,
)
from pipeline_utils import DATA_DIR, PROJECT_TO_REPO


def collect_joern_scan_targets(projects: list[str]) -> list[tuple[str, str]]:
    """
    Walk the data directory and collect (srcDir, outDir) pairs.
    outDir is srcDir/joern/ — where genJoern.sc writes the dot files.
    Only includes directories containing at least one .c file.
    """
    targets: list[tuple[str, str]] = []
    for project in projects:
        project_data_dir = Path(DATA_DIR) / project
        if not project_data_dir.is_dir():
            print(f"[WARN] data dir not found, skipping: {project_data_dir}")
            continue
        for variant in ("before", "after", "fixing"):
            for src_dir in sorted(project_data_dir.rglob(variant)):
                if src_dir.is_dir() and any(src_dir.rglob("*.c")):
                    out_dir = src_dir / "joern"
                    targets.append((str(src_dir), str(out_dir)))
    return targets

def run(projects: list[str]) -> None:
    joern_script = get_joern_script()
    targets_file = get_joern_targets_file()
    mapping_dir  = targets_file.parent   # graph_construction/mapping/

    targets = collect_joern_scan_targets(projects)
    if not targets:
        print("No targets found — nothing to do.")
        return

    # Write targets file: genJoern.sc expects "srcDir;outDir" per line
    targets_file.parent.mkdir(parents=True, exist_ok=True)
    targets_file.write_text(
        "\n".join(f"{src};{out}" for src, out in targets) + "\n",
        encoding="utf-8"
    )
    print(f"Wrote {len(targets)} targets to {targets_file}")

    # Run joern once from mapping/ so genJoern.sc finds joern_targets.txt there
    cmd = ["joern", "--script", str(joern_script)]
    print(f"Running: {' '.join(cmd)}")
    print(f"CWD    : {mapping_dir}")

    result = subprocess.run(cmd, cwd=str(mapping_dir))
    if result.returncode != 0:
        print(f"[ERROR] Joern exited with code {result.returncode}")
    else:
        print("[OK] Joern completed")


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Joern CPG dot files for before/after/fixing source directories."
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=get_all_projects(),
        help="Projects to process (default: all configured projects).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    run(projects=args.projects)