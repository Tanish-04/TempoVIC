"""
config_loader.py
================
Loads graph_construction/config.yaml and exposes typed getter functions.

Usage:
    from config_loader import get_project_root, get_project_compile_db, ...

The YAML file is loaded once and cached for the lifetime of the process.
Call reload_config() to force a fresh read (useful in tests).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_FILE_PATH = CONFIG_DIR / "config.yaml"
cached_config: Optional[Dict[str, Any]] = None


# ── Internal helpers ──────────────────────────────────────────────────────────

def load_config() -> Dict[str, Any]:
    global cached_config
    if cached_config is None:
        if not CONFIG_FILE_PATH.is_file():
            raise FileNotFoundError(
                f"Config file not found: {CONFIG_FILE_PATH}\n"
                "Place config.yaml in the same directory as config_loader.py."
            )
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        if not isinstance(raw, dict):
            raise ValueError(
                f"config.yaml must be a mapping at the top level, got {type(raw)}"
            )
        cached_config = raw
    return cached_config


def reload_config() -> None:
    """Clear the cache so the next call re-reads config.yaml from disk."""
    global cached_config
    cached_config = None


def resolve_config_path(value: str) -> Path:
    """
    Return an absolute Path.
    Absolute values are used as-is; relative values are resolved
    relative to the directory that contains config.yaml.
    """
    p = Path(value).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (CONFIG_DIR / p).resolve()


# ── Clang ─────────────────────────────────────────────────────────────────────

def get_libclang_path() -> str:
    """Absolute path to libclang.so — passed to clang.cindex.conf.set_library_file."""
    cfg = load_config()["clang"]
    env_var = str(cfg.get("libclang_path_env_var", "TEMPOVC_LIBCLANG"))
    override = os.environ.get(env_var, "").strip()
    if override:
        p = Path(override).expanduser().resolve()
    else:
        p = resolve_config_path(cfg["libclang_path"])
    if not p.is_file():
        raise FileNotFoundError(f"libclang not found at configured path: {p}")
    return str(p)


def get_c_standard() -> str:
    """C standard string, e.g. 'c99'. Inserted as -std=<value> in compile args."""
    return str(load_config()["clang"]["c_standard"])


def get_system_include_paths() -> List[str]:
    """System include directories appended as -I flags after compile-db flags."""
    cfg = load_config().get("clang") or {}
    paths = cfg.get("system_include_paths") or ["/usr/include", "/usr/local/include"]
    if not isinstance(paths, list):
        raise TypeError("clang.system_include_paths must be a list of strings")
    return [
        str(Path(p).expanduser()) if Path(p).is_absolute()
        else str(resolve_config_path(p))
        for p in paths
    ]


def get_fake_libc_include() -> str:
    """
    Path to fake libc headers used by the parser generator.
    Caller is responsible for prefixing '-I' if needed.
    """
    return str(resolve_config_path(load_config()["clang"]["fake_libc_include"]))


# ── GumTree ───────────────────────────────────────────────────────────────────

def get_gumtree_java_home() -> str:
    """Absolute path to the Java 21 home directory required by GumTree."""
    return str(resolve_config_path(load_config()["gumtree"]["java_home"]))


def get_gumtree_bin_path() -> Path:
    """
    Absolute Path to the GumTree executable.
    bin_relative_path in config.yaml is resolved relative to mapping/.
    """
    cfg = load_config()["gumtree"]
    rel = Path(cfg["bin_relative_path"])
    p = rel.resolve() if rel.is_absolute() else (CONFIG_DIR / "mapping" / rel).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"GumTree binary not found: {p}")
    return p


def get_gumtree_timeout() -> int:
    """Subprocess timeout in seconds for GumTree calls."""
    return int(load_config()["gumtree"]["timeout_seconds"])


# ── Project selection ─────────────────────────────────────────────────────────

def get_active_project() -> str:
    """
    Returns the currently active project name.
    Reads the env variable named in config (default: TEMPOVC_PROJECT).
    Falls back to default_project in config.yaml if the env var is unset.
    """
    cfg = load_config()
    env_var = str(cfg.get("project_env_var", "TEMPOVC_PROJECT"))
    name = os.environ.get(env_var, "").strip()
    if not name:
        name = str(cfg.get("default_project", "linux"))
    return name


def project_config(project: str = None) -> Dict[str, Any]:
    """
    Returns the config block for the named project, or the active one if None.
    Raises KeyError with a clear message when the project is not defined.
    """
    name = project or get_active_project()
    projects = load_config().get("projects", {})
    if name not in projects:
        raise KeyError(
            f"Project '{name}' not found in config.yaml. "
            f"Available projects: {list(projects.keys())}"
        )
    return projects[name]


# ── Project-specific getters ──────────────────────────────────────────────────

def get_project_root(project: str = None) -> str:
    """
    Root directory of the target project's source tree.
    Pass a project name explicitly, or leave None to use the active project.
    """
    return str(resolve_config_path(project_config(project)["root"]))


def get_project_compile_db(project: str = None) -> str:
    """
    Resolves compile_commands.json for the active (or named) project:
      1. If the per-project compile_db_env_var is set in the environment, use it.
      2. Otherwise fall back to the compile_db path in config.yaml.
    """
    cfg = project_config(project)
    env_var = str(cfg.get("compile_db_env_var", "TEMPOVC_COMPILE_DB"))
    override = os.environ.get(env_var, "").strip()
    if override:
        return str(Path(override).expanduser().resolve())
    return str(resolve_config_path(cfg["compile_db"]))

def get_repos_dir() -> str:
    """Root directory where all project git repositories are cloned."""
    return str(resolve_config_path(load_config()["pipeline"]["repos_dir"]))


def get_data_dir() -> str:
    """Root directory for generalizability pipeline output data."""
    return str(resolve_config_path(load_config()["pipeline"]["data_dir"]))


def get_project_repo_name(project: str = None) -> str:
    """
    Returns the repository folder name for a project.
    e.g. get_project_repo_name('FFmpeg') -> 'FFmpeg'
    Falls back to the project key itself if repo_name is not set.
    """
    cfg = project_config(project)
    return str(cfg.get("repo_name", project or get_active_project()))


def get_all_projects() -> list:
    """Returns the list of all project names defined in config.yaml."""
    return list(load_config().get("projects", {}).keys())

def get_joern_script() -> Path:
    """Absolute Path to the Joern .sc script used to generate CPG graphs."""
    p = resolve_config_path(load_config()["joern"]["script_path"])
    if not p.is_file():
        raise FileNotFoundError(f"Joern script not found: {p}")
    return p


def get_joern_targets_file() -> Path:
    """Absolute Path to the Joern targets file listing source directories."""
    return resolve_config_path(load_config()["joern"]["targets_file"])

def get_pyszz_path() -> Path:
    """Absolute Path to the pyszz directory (added to sys.path for V-SZZ imports)."""
    p = resolve_config_path(load_config()["vszz"]["pyszz_path"])
    if not p.is_dir():
        raise FileNotFoundError(f"pyszz directory not found: {p}")
    return p