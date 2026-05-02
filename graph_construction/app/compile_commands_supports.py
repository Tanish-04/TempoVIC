import json
import os
import shlex
from typing import Dict, List

from config_loader import get_project_root

KERNEL_ROOT = get_project_root()

compile_db_cache = None
compile_db_loaded_path = None
compile_db_by_path: Dict[str, List[dict]] = {}
compile_db_by_sanitized: Dict[str, List[dict]] = {}


def ensure_compile_db_loaded(compile_db_path: str) -> None:
    global compile_db_cache, compile_db_loaded_path, compile_db_by_path, compile_db_by_sanitized

    if compile_db_loaded_path == compile_db_path:
        return

    compile_db_cache = None
    compile_db_by_path.clear()
    compile_db_by_sanitized.clear()

    if not os.path.exists(compile_db_path):
        raise FileNotFoundError(f"compile_commands.json not found: {compile_db_path}")

    with open(compile_db_path, "r") as f:
        compile_db_cache = json.load(f)

    compile_db_anchor = os.path.dirname(os.path.abspath(compile_db_path))
    repo_root = compile_db_anchor

    print(f"[COMPILE-DB] Repo anchor (sanitized keys / path resolution): {repo_root}")

    for entry in compile_db_cache:
        file_path = entry.get("file")
        if not file_path:
            continue

        directory = entry.get("directory") or "."
        if not os.path.isabs(directory):
            base_dir = os.path.normpath(os.path.join(compile_db_anchor, directory))
        else:
            base_dir = os.path.normpath(directory)

        if not os.path.isabs(file_path):
            file_path = os.path.normpath(os.path.join(base_dir, file_path))
        else:
            file_path = os.path.normpath(os.path.abspath(file_path))

        abs_path = os.path.normpath(os.path.abspath(file_path))
        entry["resolved_source_path"] = abs_path
        compile_db_by_path.setdefault(abs_path, []).append(entry)

        try:
            rel_path = os.path.relpath(abs_path, repo_root)
        except ValueError:
            rel_path = abs_path

        sanitized = rel_path.replace(os.sep, "_")
        compile_db_by_sanitized.setdefault(sanitized, []).append(entry)

    compile_db_loaded_path = compile_db_path
    print(f"[COMPILE-DB] Loaded {len(compile_db_cache)} entries from {compile_db_path}")


def generate_kernel_path_candidates(base_dir: str, rest: str):
    """
    Generate candidate kernel paths by selectively converting underscores to '/'
    in the remainder string. Prefers candidates with more directory separators.
    """
    if not rest:
        yield os.path.normpath(os.path.join(KERNEL_ROOT, base_dir))
        return

    positions = [idx for idx, ch in enumerate(rest) if ch == "_"]
    if not positions:
        yield os.path.normpath(os.path.join(KERNEL_ROOT, base_dir, rest))
        return

    def popcount(x: int) -> int:
        return bin(x).count("1")

    seen = set()
    masks = sorted(range(1 << len(positions)), key=lambda m: (-popcount(m), m))
    for mask in masks:
        chars = list(rest)
        for bit, pos in enumerate(positions):
            if mask & (1 << bit):
                chars[pos] = os.sep
        candidate_rel = "".join(chars)
        candidate_path = os.path.normpath(os.path.join(KERNEL_ROOT, base_dir, candidate_rel))
        if candidate_path not in seen:
            seen.add(candidate_path)
            yield candidate_path


def extract_compile_arguments(entry: dict) -> List[str]:
    """Extract compiler arguments from a compile_commands.json entry."""
    cmd = entry.get("command") or entry.get("arguments") or []
    if isinstance(cmd, str):
        tokens = shlex.split(cmd)
    elif isinstance(cmd, list):
        tokens = list(cmd)
    else:
        tokens = []
    return [tok for tok in tokens if not tok.startswith("gcc") and not tok.startswith("-o")]


def get_compile_args_from_db(source_file: str, compile_db_path: str) -> list:
    """
    Fetch exact compile arguments for a source file from compile_commands.json.
    Falls back to [] if not found.
    """
    if not os.path.exists(compile_db_path):
        print(f"[WARN] compile_commands.json not found at {compile_db_path}")
        return []
    try:
        compile_db_anchor = os.path.dirname(os.path.abspath(compile_db_path))
        want = os.path.normpath(os.path.abspath(source_file))
        with open(compile_db_path, "r") as f:
            db = json.load(f)
        for entry in db:
            file_path = entry.get("file")
            if not file_path:
                continue
            directory = entry.get("directory") or "."
            if not os.path.isabs(directory):
                base_dir = os.path.normpath(os.path.join(compile_db_anchor, directory))
            else:
                base_dir = os.path.normpath(directory)
            if not os.path.isabs(file_path):
                abs_file = os.path.normpath(os.path.join(base_dir, file_path))
            else:
                abs_file = os.path.normpath(os.path.abspath(file_path))
            if abs_file != want:
                continue
            cmd = entry.get("command")
            if cmd:
                tokens = shlex.split(cmd)
            else:
                tokens = list(entry.get("arguments") or [])
            args = [a for a in tokens if not a.startswith("gcc") and not a.startswith("-o")]
            print(f"[COMPILE-DB] Found {len(args)} args for {os.path.basename(source_file)}")
            return args
    except Exception as e:
        print(f"[WARN] Could not parse compile_commands.json: {e}")
    return []
