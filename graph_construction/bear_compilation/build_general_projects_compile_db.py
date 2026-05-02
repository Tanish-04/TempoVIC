#!/usr/bin/env python3
"""
Generate compile_commands.json for each non-Linux project (FFmpeg, OpenSSL,
ImageMagick, PHP-SRC) using Bear.

Usage:
  python3 build_general_projects_compile_db.py                    # all four projects
  python3 build_general_projects_compile_db.py FFmpeg             # one project
  python3 build_general_projects_compile_db.py FFmpeg OpenSSL     # subset

Requirements: bear, make, gcc/clang, each project's build dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

graph_construction_root = Path(__file__).resolve().parent.parent
if str(graph_construction_root) not in sys.path:
    sys.path.insert(0, str(graph_construction_root))

from config_loader import get_project_compile_db, get_repos_dir


def log(msg: str) -> None:
    print(f"[INFO]  {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN]  {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[FAIL]  {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def tail_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def run_command(
    cmd: Sequence[str],
    cwd: Path,
    *,
    check: bool = True,
    log_tail: int = 0,
) -> subprocess.CompletedProcess[str]:
    r = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    combined = (r.stdout or "") + (r.stderr or "")
    if log_tail and combined.strip():
        log(tail_lines(combined, log_tail))
    if check and r.returncode != 0:
        if not log_tail and combined.strip():
            print(combined, file=sys.stderr, end="")
        raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
    return r


def check_bear() -> None:
    if shutil.which("bear") is None:
        fail("bear not found. Install with: sudo apt install bear")
    r = subprocess.run(["bear", "--version"], capture_output=True, text=True)
    line = (r.stdout or r.stderr or "").splitlines()[0] if (r.stdout or r.stderr) else "(no output)"
    log(f"Bear version: {line}")


def verify_compile_db(repo_dir: Path, project: str) -> None:
    """
    Bear leaves compile_commands.json in the repo cwd. That file must match
    config.yaml's projects.<project>.compile_db (used by get_project_compile_db).

    Note: if TEMPOVC_COMPILE_DB is set in the environment it overrides every
    project's compile_db path in config_loader; unset it when building multiple
    repos so each project's YAML path is used.
    """
    produced = (repo_dir / "compile_commands.json").resolve()
    expected = Path(get_project_compile_db(project)).resolve()

    if not produced.is_file():
        warn(f"{project}: compile_commands.json not found at {produced}")
        raise FileNotFoundError(str(produced))

    if produced != expected:
        fail(
            f"{project}: compile_commands.json is at {produced} but config.yaml "
            f"expects {expected}. Set matching paths or TEMPOVC_COMPILE_DB."
        )

    try:
        data = json.loads(produced.read_text(encoding="utf-8"))
        count = len(data) if isinstance(data, list)
    except (json.JSONDecodeError, OSError):
        count = "?"
    log(f"{project}: compile_commands.json OK — {produced} ({count} entries)")


def cpu_parallelism() -> int:
    return os.cpu_count() or 4


def build_ffmpeg(repos_dir: Path) -> None:
    repo_dir = repos_dir / "FFmpeg"
    log("=== FFmpeg ===")
    if not repo_dir.is_dir():
        fail(f"FFmpeg repo not found: {repo_dir}")

    run_command(["make", "distclean"], cwd=repo_dir, check=False)
    log("Configuring FFmpeg (disable non-essential components for speed)...")
    run_command(
        [
            str(repo_dir / "configure"),
            "--disable-optimizations",
            "--disable-stripping",
            "--enable-debug",
            "--disable-x86asm",
            "--cc=gcc",
        ],
        cwd=repo_dir,
        log_tail=5,
    )
    log("Running Bear + make...")
    run_command(["bear", "--", "make", f"-j{cpu_parallelism()}"], cwd=repo_dir, log_tail=10)
    verify_compile_db(repo_dir, "FFmpeg")


def build_imagemagick(repos_dir: Path) -> None:
    repo_dir = repos_dir / "ImageMagick"
    log("=== ImageMagick ===")
    if not repo_dir.is_dir():
        fail(f"ImageMagick repo not found: {repo_dir}")

    run_command(["make", "distclean"], cwd=repo_dir, check=False)
    log("Configuring ImageMagick...")
    run_command(
        [str(repo_dir / "configure"), "--enable-debug", "--without-x"],
        cwd=repo_dir,
        log_tail=5,
    )
    log("Running Bear + make...")
    run_command(["bear", "--", "make", f"-j{cpu_parallelism()}"], cwd=repo_dir, log_tail=10)
    verify_compile_db(repo_dir, "ImageMagick")


def build_openssl(repos_dir: Path) -> None:
    repo_dir = repos_dir / "OpenSSL"
    log("=== OpenSSL ===")
    if not repo_dir.is_dir():
        fail(f"OpenSSL repo not found: {repo_dir}")

    run_command(["make", "distclean"], cwd=repo_dir, check=False)
    log("Configuring OpenSSL...")
    run_command(
        [str(repo_dir / "Configure"), "linux-x86_64", "--debug", "no-shared"],
        cwd=repo_dir,
        log_tail=5,
    )
    log("Running Bear + make...")
    run_command(["bear", "--", "make", f"-j{cpu_parallelism()}"], cwd=repo_dir, log_tail=10)
    verify_compile_db(repo_dir, "OpenSSL")


def build_php(repos_dir: Path) -> None:
    repo_dir = repos_dir / "PHP-SRC"
    log("=== PHP-SRC ===")
    if not repo_dir.is_dir():
        fail(f"PHP-SRC repo not found: {repo_dir}")

    run_command(["make", "distclean"], cwd=repo_dir, check=False)
    log("Running buildconf...")
    run_command([str(repo_dir / "buildconf"), "--force"], cwd=repo_dir, log_tail=3)
    log("Configuring PHP...")
    run_command(
        [str(repo_dir / "configure"), "--enable-debug", "--disable-all"],
        cwd=repo_dir,
        log_tail=5,
    )
    log("Running Bear + make...")
    run_command(["bear", "--", "make", f"-j{cpu_parallelism()}"], cwd=repo_dir, log_tail=10)
    verify_compile_db(repo_dir, "PHP-SRC")


BUILDERS: Dict[str, Callable[[Path], None]] = {
    "FFmpeg": build_ffmpeg,
    "ImageMagick": build_imagemagick,
    "OpenSSL": build_openssl,
    "PHP-SRC": build_php,
}

DEFAULT_PROJECTS: Tuple[str, ...] = ("FFmpeg", "ImageMagick", "OpenSSL", "PHP-SRC")


def main(argv: Sequence[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    graph_root = script_dir.parent.resolve()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        type=Path,
        default=graph_root,
        help=f"graph_construction root (contains repositories/). Default: {graph_root}",
    )
    ap.add_argument(
        "projects",
        nargs="*",
        metavar="PROJECT",
        help="Subset: FFmpeg, ImageMagick, OpenSSL, PHP-SRC (default: all four)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    base_dir: Path = args.base.resolve()
    if base_dir == graph_root:
        repos_dir = Path(get_repos_dir())
    else:
        repos_dir = (base_dir / "repositories").resolve()

    projects: List[str] = list(args.projects) if args.projects else list(DEFAULT_PROJECTS)

    check_bear()
    log(f"Base: {base_dir}")
    log(f"Repos: {repos_dir} (must match config.yaml pipeline.repos_dir + project roots)")
    log(f"Projects to build: {' '.join(projects)}")
    print("", flush=True)

    ok = 0
    fail_count = 0
    for name in projects:
        builder = BUILDERS.get(name)
        if builder is None:
            warn(f"Unknown project: {name} — skipping")
            print("", flush=True)
            continue
        try:
            builder(repos_dir)
            ok += 1
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            warn(f"{name}: {e}")
            fail_count += 1
        print("", flush=True)

    print("BEAR COMPLETE")
    print(f"  Built ok : {ok}")
    print(f"  Failed   : {fail_count}")
    print("compile_commands.json for each project is under repositories/<Project>/")
    print("(same paths as config.yaml compile_db). Pipeline: set TEMPOVC_PROJECT, then")
    print("generate_graphs.py / Clang use get_project_compile_db() automatically.")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
