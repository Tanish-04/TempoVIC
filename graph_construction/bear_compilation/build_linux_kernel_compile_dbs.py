"""
Builds compile_commands.json for every required Linux kernel version and architecture, then merges them into a single unified database.

Usage 
    python bear_compilation/build_linux_kernel_compile_dbs.py               
    python bear_compilation/build_linux_kernel_compile_dbs.py --only x86 arm64 v4.19
    python bear_compilation/build_linux_kernel_compile_dbs.py --merge-only
    python bear_compilation/build_linux_kernel_compile_dbs.py --force x86   # rebuild even if done
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional



HERE = Path(__file__).resolve().parent          # bear_compilation/
ROOT = HERE.parent                              # graph_construction/

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import get_project_compile_db, get_project_root

# Linux clone location and merged compile_commands.json — from config.yaml projects.linux
KERNEL = Path(get_project_root("linux"))
MERGE_OUT = Path(get_project_compile_db("linux"))
LOGS = HERE / "build_logs"


@dataclass
class BuildSpec:
    id: str
    description: str
    output: str           # filename relative to bear_compilation/
    git_tag: Optional[str]
    arch: Optional[str]
    cross_compile: Optional[str]
    config_target: str    # "allyesconfig" | "defconfig"
    extra_config: List[str] = field(default_factory=list)
    # Keys for compat_fixes dict below
    compat_fixes: List[str] = field(default_factory=list)
    min_entries: int = 10_000
    gcc: str = "gcc"
    kcflags: str = ""
    keep_going: bool = False


# Compatibility fix recipes
COMPAT_FIX_COMMANDS = {
    # Remove -Werror from specific build-tool Makefiles
    "werror": [
        "sed -i 's/-Werror//g'"
        " tools/build/feature/Makefile"
        " tools/lib/subcmd/Makefile"
        " tools/objtool/Makefile"
        " 2>/dev/null || true",
    ],
    # Remove -Werror from ALL Makefiles (for very old kernels)
    "werror-all": [
        r"find . -name 'Makefile*' -exec sed -i 's/-Werror//g' {} \; 2>/dev/null || true",
    ],
    # Suppress use-after-free GCC warning that kills old-kernel builds
    "use-after-free": [
        r"sed -i '1a\#pragma GCC diagnostic push\n#pragma GCC diagnostic ignored \"-Wuse-after-free\"'"
        r" tools/lib/subcmd/subcmd-util.h 2>/dev/null || true",
    ],
    # Fix duplicate yylloc symbol in dtc lexer
    "yylloc": [
        "sed -i 's/YYLTYPE yylloc;/extern YYLTYPE yylloc;/'"
        " scripts/dtc/dtc-lexer.l"
        " scripts/dtc/dtc-lexer.lex.c"
        " 2>/dev/null || true",
    ],
    "make4-compat": [
        r"""cat > /tmp/_kernelfix.py << 'PYEOF'
    import re
    with open('Makefile', 'r') as f:
        lines = f.readlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(\S+) (%\S+):(.*)$', line)
        if m:
            normal, pattern, rest = m.group(1), m.group(2), m.group(3)
            recipe = []
            j = i + 1
            while j < len(lines):
                if lines[j].startswith('\t'):
                    recipe.append(lines[j])
                    j += 1
                elif recipe and recipe[-1].rstrip().endswith('\\'):
                    recipe.append(lines[j])
                    j += 1
                else:
                    break
            out.append(f'{normal}:{rest}\n')
            out.extend(recipe)
            out.append(f'{pattern}:{rest}\n')
            out.extend(recipe)
            i = j
        else:
            out.append(line)
            i += 1
    with open('Makefile', 'w') as f:
        f.writelines(out)
    PYEOF
    python3 /tmp/_kernelfix.py""",
        ],
    }


BUILDS: List[BuildSpec] = [
    BuildSpec(
        id="x86",
        description="Current kernel x86_64 – allyesconfig",
        output="compile_commands_x86.json",
        git_tag=None,
        arch=None, cross_compile=None,
        config_target="allyesconfig",
        min_entries=20_000,
        gcc="gcc",
    ),
    BuildSpec(
        id="arm64",
        description="Current kernel ARM64 – allyesconfig",
        output="compile_commands_arm64.json",
        git_tag=None,
        arch="arm64", cross_compile="aarch64-linux-gnu-",
        config_target="allyesconfig",
        min_entries=15_000,
        gcc="gcc",
    ),
    BuildSpec(
        id="arm",
        description="Current kernel ARM 32-bit – allyesconfig",
        output="compile_commands_arm.json",
        git_tag=None,
        arch="arm", cross_compile="arm-linux-gnueabihf-",
        config_target="allyesconfig",
        min_entries=12_000,
        gcc="gcc",
    ),
    BuildSpec(
        id="s390",
        description="Current kernel s390 – allyesconfig",
        output="compile_commands_s390.json",
        git_tag=None,
        arch="s390", cross_compile="s390x-linux-gnu-",
        config_target="allyesconfig",
        min_entries=12_000,
        gcc="gcc",
    ),
    BuildSpec(
        id="powerpc",
        description="Current kernel PowerPC – allyesconfig",
        output="compile_commands_powerpc.json",
        git_tag=None,
        arch="powerpc", cross_compile="powerpc-linux-gnu-",
        config_target="allyesconfig",
        min_entries=15_000,
        gcc="gcc",
    ),
    BuildSpec(
        id="v5.10",
        description="Linux v5.10 LTS x86_64 – allyesconfig",
        output="compile_commands_v5.10_x86.json",
        git_tag="v5.10",
        arch=None, cross_compile=None,
        config_target="allyesconfig",
        extra_config=[
            "--disable SECURITY_SELINUX",
            "--disable RETPOLINE",
            "--enable IO_URING",
            "--enable CRYPTO_USER",
            "--enable PSI",
        ],
        compat_fixes=["werror", "use-after-free"],
        min_entries=15_000,
        gcc="gcc-11",
        keep_going=True,         
    ),
    BuildSpec(
        id="v4.19",
        description="Linux v4.19 LTS x86_64 – allyesconfig",
        output="compile_commands_v4.19_x86.json",
        git_tag="v4.19",
        arch=None, cross_compile=None,
        config_target="allyesconfig",
        extra_config=[
            "--disable SECURITY_SELINUX",
            "--disable RETPOLINE",
            "--disable HYPERV",
            "--disable NFP",
        ],
        compat_fixes=["werror", "use-after-free", "yylloc"],
        min_entries=12_000,
        gcc="gcc-11",
        keep_going=True,        
    ),
    BuildSpec(
        id="v2.6.30",
        description="Linux v2.6.30 x86_64 – defconfig + media/staging",
        output="compile_commands_v2.6.30_x86.json",
        git_tag="v2.6.30",
        arch=None, cross_compile=None,
        config_target="defconfig",
        extra_config=[
            "--enable MEDIA_SUPPORT",
            "--enable VIDEO_DEV",
            "--enable DVB_CORE",
            "--enable STAGING",
            "--enable INOTIFY_USER",
            "--disable SECURITY_SELINUX",
            "--disable DEBUG_INFO",
        ],
        compat_fixes=["make4-compat", "werror-all", "yylloc"],  
        min_entries=5_000,
        gcc="gcc-11",
        keep_going=True,
        kcflags="-Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=incompatible-pointer-types -Wno-error=strict-prototypes -Wno-error=return-type",
    ),
    BuildSpec(
        id="v2.6.24",
        description="Linux v2.6.24 x86_64 – defconfig + KVM/block",
        output="compile_commands_v2.6.24_x86.json",
        git_tag="v2.6.24",
        arch=None, cross_compile=None,
        config_target="defconfig",
        extra_config=[
            "--enable VIRTUALIZATION",
            "--enable KVM",
            "--enable KVM_INTEL",
            "--enable KVM_AMD",
            "--enable BLK_DEV_IO_TRACE",
            "--disable SECURITY_SELINUX",
            "--disable DEBUG_INFO",
        ],
        compat_fixes=["make4-compat", "werror-all"],
        min_entries=3_000,
        gcc="gcc-11",
        keep_going=True,
        kcflags="-Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=incompatible-pointer-types -Wno-error=strict-prototypes -Wno-error=return-type -Wno-error=old-style-definition",
    ),
]



saved_head: Optional[str] = None
git_dirty = False


def git_cmd(args: List[str], cwd: Path = KERNEL) -> str:
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip()


def save_git_head() -> None:
    global saved_head, git_dirty
    saved_head = git_cmd(["rev-parse", "HEAD"])
    dirty = git_cmd(["status", "--porcelain"])
    git_dirty = bool(dirty)
    if git_dirty:
        print("[GIT] Stashing uncommitted changes …")
        git_cmd(["stash", "push", "-m", "build_linux_kernel_compile_dbs_autostash"])


def restore_git_head() -> None:
    if saved_head is None:
        return
    # Clean up any stale index.lock before restoring
    remove_index_lock()
    current = git_cmd(["rev-parse", "HEAD"])
    if current != saved_head:
        print(f"\n[GIT] Restoring HEAD → {saved_head[:12]} …")
        git_cmd(["checkout", "--", "."])          # discard local changes from patches
        git_cmd(["checkout", saved_head])
    if git_dirty:
        stashes = git_cmd(["stash", "list"])
        if "build_linux_kernel_compile_dbs_autostash" in stashes:
            print("[GIT] Popping stash …")
            git_cmd(["stash", "pop"])
    print("[GIT] Kernel repo restored to original state.")


def remove_index_lock() -> None:
    """Remove stale .git/index.lock if present."""
    lock = KERNEL / ".git" / "index.lock"
    if lock.exists():
        print(f"[GIT] Removing stale {lock}")
        lock.unlink()


# Build status helpers
def entry_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            return len(json.load(f))
    except Exception:
        return 0

def build_is_done(spec: BuildSpec) -> bool:
    out = HERE / spec.output
    return entry_count(out) >= spec.min_entries


# Status display
def cmd_status() -> None:
    print(f"\n{'ID':<10} {'STATUS':<10} {'ENTRIES':>8}  {'MIN':>8}  DESCRIPTION")
    print("─" * 78)
    total = 0
    for s in BUILDS:
        out = HERE / s.output
        count = entry_count(out)
        total += count
        done = count >= s.min_entries
        status = "done" if done else ("partial" if count > 0 else "missing")
        print(f"{s.id:<10} {status:<10} {count:>8,}  {s.min_entries:>8,}  {s.description}")
    print("─" * 78)
    print(f"{'TOTAL':<10} {'':10} {total:>8,}")

    merged_count = entry_count(MERGE_OUT)
    print(f"\nActive compile_commands.json  →  {MERGE_OUT}")
    print(f"Current entry count           →  {merged_count:,}")
    print()



def run_build(spec: BuildSpec, log_fh) -> bool:
    """
    Execute one build. Returns True on success, False on failure.
    All output is tee'd to log_fh and stdout.
    """

    def log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        log_fh.write(line + "\n")
        log_fh.flush()

    kernel = KERNEL
    out_in_kernel = kernel / spec.output
    out_in_bear   = HERE / spec.output

    log(f"=== BUILD: {spec.id}  ({spec.description}) ===")

    lock = kernel / ".git" / "index.lock"
    if lock.exists():
        log(f"Removing stale {lock}")
        lock.unlink()

    # 1. Checkout git tag if needed
    if spec.git_tag:
        log(f"git checkout {spec.git_tag} …")
        r = subprocess.run(
            ["git", "checkout", "--", "."],
            cwd=kernel, capture_output=True, text=True
        )
        r = subprocess.run(
            ["git", "checkout", spec.git_tag],
            cwd=kernel, capture_output=True, text=True
        )
        if r.returncode != 0:
            log(f"ERROR: git checkout failed:\n{r.stderr}")
            return False

    # 2. make mrproper (BEFORE compat fixes so fixes aren't wiped)
    log("make mrproper …")
    subprocess.run(["make", "mrproper"], cwd=kernel,
                   stdout=log_fh, stderr=log_fh)

    # 3. Compatibility fixes (AFTER mrproper so they survive)
    for fix_key in spec.compat_fixes:
        for cmd in COMPAT_FIX_COMMANDS.get(fix_key, []):
            log(f"  compat fix [{fix_key}]: {cmd[:80]}")
            subprocess.run(cmd, shell=True, cwd=kernel)

    # 4. Set arch / cross-compile env
    env = os.environ.copy()
    if spec.arch:
        env["ARCH"] = spec.arch
        log(f"ARCH={spec.arch}")
    else:
        env.pop("ARCH", None)

    if spec.cross_compile:
        env["CROSS_COMPILE"] = spec.cross_compile
        log(f"CROSS_COMPILE={spec.cross_compile}")
    else:
        env.pop("CROSS_COMPILE", None)

    # 5. make <config_target>
    log(f"make {spec.config_target} …")
    r = subprocess.run(
        ["make", spec.config_target],
        cwd=kernel, env=env, stdout=log_fh, stderr=log_fh
    )
    if r.returncode != 0:
        log(f"ERROR: make {spec.config_target} failed (rc={r.returncode})")
        return False

    # 6. Extra config options
    for opt in spec.extra_config:
        log(f"  scripts/config {opt}")
        subprocess.run(
            f"scripts/config {opt}",
            shell=True, cwd=kernel, env=env,
            stdout=log_fh, stderr=log_fh
        )

    if spec.extra_config:
        log("make olddefconfig …")
        r = subprocess.run(
            ["make", "olddefconfig"],
            cwd=kernel, env=env, stdout=log_fh, stderr=log_fh
        )
        if r.returncode != 0:
            log("olddefconfig unavailable (pre-v3.7 kernel) – "
                "falling back to: yes '' | make oldconfig")
            subprocess.run(
                "yes '' | make oldconfig",
                shell=True, cwd=kernel, env=env,
                stdout=log_fh, stderr=log_fh
            )

    # 7. bear + make
    ncpu = os.cpu_count() or 4
    # For cross-arch builds, use the cross-compiler; for native, use spec.gcc
    if spec.cross_compile:
        cc = f"{spec.cross_compile}{spec.gcc}"
    else:
        cc = spec.gcc
    make_cmd = ["make", f"CC={cc}", f"-j{ncpu}"]
    if spec.keep_going:
        make_cmd.append("-k")
    if spec.kcflags:
        make_cmd.append(f"KCFLAGS={spec.kcflags}")
    bear_cmd = ["bear", "--output", str(out_in_kernel), "--"] + make_cmd

    log(f"bear + {' '.join(make_cmd)}  (logging to {log_fh.name}) …")
    t0 = time.time()

    if spec.kcflags:
        env["KCFLAGS"] = spec.kcflags

    proc = subprocess.Popen(
        bear_cmd, cwd=kernel, env=env,
        stdout=log_fh, stderr=subprocess.STDOUT
    )
    proc.wait()
    elapsed = time.time() - t0
    log(f"Build finished in {elapsed/60:.1f} min  (bear exit={proc.returncode})")

    # 8. Verify output
    count = entry_count(out_in_kernel)
    log(f"Entries in {spec.output}: {count:,}  (min required: {spec.min_entries:,})")

    if count < spec.min_entries:
        log(f"WARNING: entry count {count} < minimum {spec.min_entries}. "
            "Build may have failed at linking — check log. "
            "Keeping partial output anyway.")

    if count == 0:
        log("ERROR: no entries produced — skipping copy.")
        return False

    # 9. Copy to bear_compilation/
    shutil.copy2(out_in_kernel, out_in_bear)
    log(f"Copied → {out_in_bear}")
    return True

def cmd_merge() -> None:
    shards = []
    for spec in BUILDS:
        p = HERE / spec.output
        if p.exists() and entry_count(p) > 0:
            shards.append(p)

    if not shards:
        print("ERROR: no compile_commands_*.json shards found in bear_compilation/")
        sys.exit(1)

    print(f"\nMerging {len(shards)} shard(s) …")
    seen: dict = {}
    total_read = 0
    for shard in shards:
        with open(shard) as f:
            entries = json.load(f)
        total_read += len(entries)
        for e in entries:
            key = (os.path.normpath(e.get("file", "")), e.get("directory", ""))
            seen[key] = e
        print(f"  {len(entries):>7,} entries  ← {shard.name}")

    merged = list(seen.values())
    MERGE_OUT.parent.mkdir(parents=True, exist_ok=True)

    # Write backup first
    backup = KERNEL / "compile_commands_merged.json"
    with open(backup, "w") as f:
        json.dump(merged, f, indent=1)

    # Copy as active DB
    shutil.copy2(backup, MERGE_OUT)

    print(f"\n  Total read   : {total_read:,}")
    print(f"  After dedup  : {len(merged):,}")
    print(f"  Active DB    : {MERGE_OUT}")
    print(f"  Backup       : {backup}")



def main():
    parser = argparse.ArgumentParser(
        description="Build and merge compile_commands.json for TempoVIC."
    )
    parser.add_argument("--status", action="store_true",
                        help="Show build status and exit.")
    parser.add_argument("--only", nargs="+", metavar="ID",
                        help="Build only these IDs (space-separated).")
    parser.add_argument("--force", nargs="+", metavar="ID",
                        help="Force-rebuild these IDs even if already done.")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip builds; just merge existing shards.")
    parser.add_argument("--no-merge", action="store_true",
                        help="Run builds but skip the final merge step.")
    args = parser.parse_args()

    # Validate kernel root
    if not (KERNEL / "Makefile").exists():
        print(f"ERROR: kernel source not found at {KERNEL}")
        sys.exit(1)

    LOGS.mkdir(parents=True, exist_ok=True)

    if args.status:
        cmd_status()
        return

    if args.merge_only:
        cmd_merge()
        return

    # Determine which builds to run
    valid_ids = {s.id for s in BUILDS}
    force_ids = set(args.force or [])
    only_ids  = set(args.only  or [])

    for fid in force_ids | only_ids:
        if fid not in valid_ids:
            print(f"ERROR: unknown build ID '{fid}'. Valid: {sorted(valid_ids)}")
            sys.exit(1)

    todo = [
        s for s in BUILDS
        if (not only_ids or s.id in only_ids)
        and (s.id in force_ids or not build_is_done(s))
    ]

    if not todo:
        print("All requested builds already done. Use --force <id> to rebuild.")
        cmd_status()
        if not args.no_merge:
            cmd_merge()
        return

    print(f"\nBuilds to run ({len(todo)}):")
    for s in todo:
        print(f"  • {s.id:<10}  {s.description}")

    # Save git HEAD and register restore on exit
    save_git_head()
    atexit.register(restore_git_head)

    results = {}
    for spec in todo:
        log_path = LOGS / f"{spec.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        print(f"\n{'='*70}")
        print(f"STARTING: {spec.id}  →  log: {log_path.name}")
        print(f"{'='*70}")
        with open(log_path, "w") as log_fh:
            try:
                ok = run_build(spec, log_fh)
            except KeyboardInterrupt:
                print(f"\n[INTERRUPTED] during {spec.id}")
                results[spec.id] = "interrupted"
                break
            except Exception as ex:
                print(f"\n[EXCEPTION] {spec.id}: {ex}")
                results[spec.id] = f"exception: {ex}"
                continue
        results[spec.id] = "ok" if ok else "failed"

    # Summary
    print(f"\n{'='*70}")
    print("BUILD SUMMARY")
    print(f"{'='*70}")
    for bid, status in results.items():
        icon = "✅" if status == "ok" else "❌"
        print(f"  {icon}  {bid:<10}  {status}")

    # Merge
    if not args.no_merge:
        print()
        cmd_merge()

    cmd_status()


if __name__ == "__main__":
    main()