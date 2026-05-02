# Graph Construction — TempoVIC

> End-to-end pipeline that processes Linux kernel (and other C project) CVE commits into heterogeneous graphs.
---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Environment Setup](#environment-setup)
6. [Tool Setup](#tool-setup)
   - [Bear](#bear)
   - [GumTree](#gumtree)
   - [Joern](#joern)
7. [Configuration](#configuration)
8. [Data Preparation](#data-preparation)
9. [Pipeline Steps](#pipeline-steps)
   - [Step 1 — Extract Source Files](#step-1--extract-source-files)
   - [Step 2 — Generate Joern CPG](#step-2--generate-joern-cpg)
   - [Step 3 — Build Graphs (Fix Commit)](#step-3--build-graphs-fix-commit)
   - [Step 4 — Run V-SZZ](#step-4--run-v-szz)
   - [Step 4b — Create commits.json](#step-4b--create-commitsjson)
   - [Step 5 — Extract History Source Files](#step-5--extract-history-source-files)
   - [Step 6 — Generate Joern CPG (History)](#step-6--generate-joern-cpg-history)
   - [Step 7 — Build Graphs (History Commits)](#step-7--build-graphs-history-commits)
10. [Running a Specific Project](#running-a-specific-project)
11. [Output Structure](#output-structure)
12. [Troubleshooting](#troubleshooting)

---

## Overview

The pipeline processes each CVE test case through the following stages:

```
CVE commit list (info.json)
        │
        ▼
[Step 1]  Extract before / after / fixing source files from git
        │
        ▼
[Step 2]  Run Joern to generate CPG dot files for each source directory
        │
        ▼
[Step 3]  Build heterogeneous graph.json for each fixing commit
        │
        ▼
[Step 4]  Run V-SZZ to trace deleted lines back through git history
        │
        ▼
[Step 4b] Collect all history commits into commits.json
        │
        ▼
[Step 5]  Extract source files for all history commits
        │
        ▼
[Step 6]  Run Joern on history commit source directories
        │
        ▼
[Step 7]  Build graph.json for all history commits
```

- **Steps 1–3** operate on the **fixing commit** of each CVE.
- **Steps 4–7** extend coverage to all **history commits** discovered by V-SZZ.

---

## Dataset

The fully processed dataset (ready for use without running the pipeline) is available for download:

> **[⬇ Download TempoVIC Dataset (OSF)](https://osf.io/u5s9v/overview?view_only=be083430853b4c41b56003e862995b81)**

This archive contains the complete `data/` directory with `graph.json` and `graph_vszz_full_history.json` for each test case. Download and extract it into `graph_construction/` to skip the pipeline entirely.

---

## Repository Structure

```
graph_construction/
│
├── config.yaml                        # Central config: project paths, tool paths, settings
├── config_loader.py                   # Loads config.yaml; exposes typed getter functions used by all scripts
├── pipeline_utils.py                  # Shared utilities (git helpers, file I/O) used across pipeline scripts
│
├── extract_source_files.py            # Step 1 / Step 5: Checks out before/after/fixing C source files from git
├── generate_joern.py                  # Step 2 / Step 6: Invokes Joern to generate CPG dot files for all source dirs
├── generate_graphs.py                 # Step 3 / Step 7: Builds heterogeneous graph.json from Joern CPG output
├── history_commits_retrieval.py       # Step 4: Runs V-SZZ to trace patch deletions back through git history
├── create_commits_json.py             # Step 4b: Reads V-SZZ output and writes structured commits.json per test
│
├── parse_patch/                       # Library: parses unified diff (.patch) files
│   ├── patch.py                       #   Main patch parser — reads hunks and changed lines
│   ├── patch_file.py                  #   Represents a single file changed in a patch
│   ├── patch_hunk.py                  #   Represents a hunk (contiguous block of changes) in a patch
│   └── patch_line.py                  #   Represents an individual added/deleted/context line in a hunk
│
├── parse_joerndot/                    # Library: parses Joern-generated CPG dot files into Python objects
│   ├── joern_graph.py                 #   Reads dot file and builds graph of Joern nodes and edges
│   ├── joern_node.py                  #   Represents a single CPG node (method, call, identifier, etc.)
│   └── joern_edge.py                  #   Represents a CPG edge (CFG, DFG, AST, call, etc.)
│
├── gen_finalgraph/                    # Library: assembles the final heterogeneous graph.json
│   ├── gen_graph.py                   #   Core builder — integrates patch, Joern CPG, and Clang data into one graph
│   ├── trim_graph.py                  #   Post-processes graph to prune unreachable or irrelevant nodes/edges
│   ├── node.py                        #   Node dataclass for the final output graph
│   └── edge.py                        #   Edge dataclass for the final output graph
│
├── project/                           # Library: Clang-based source file analysis
│   ├── project.py                     #   Manages a set of source files for a commit (before or after)
│   ├── source_file.py                 #   Parses a single C source file using Clang to extract functions/calls
│   ├── clang_function_tracker.py      #   Tracks function definitions and call sites via the Clang AST
│   ├── union_project.py               #   Merges before/after projects to align symbols across versions
│   ├── method.py                      #   Represents a function/method extracted from source
│   └── call.py                        #   Represents a function call site extracted from source
│
├── mapping/                           # Library: maps line numbers between before/after using GumTree AST diffs
│   ├── line_mapping.py                #   Main entry point — builds before↔after line number mapping
│   ├── gumtree_bridge.py              #   Calls GumTree binary and parses its AST edit script output
│   ├── python_parser_generator.py     #   Generates a simplified Python-parseable AST from C source
│   ├── python_parser_visitor.py       #   AST visitor that extracts token positions for GumTree input
│   ├── get_pos.py                     #   Utility to resolve line/column positions within source text
│   ├── genJoern.sc                    #   Joern Scala script: runs cpg2dot on all entries in joern_targets.txt
│   └── gumtree/                       #   GumTree tool binary (built from source; see Tool Setup)
│
├── bear_compilation/                  # Scripts to generate compile_commands.json via Bear
│   ├── build_linux_kernel_compile_dbs.py   # Builds Linux kernel across architectures and merges compile DBs
│   ├── build_general_projects_compile_db.py # Builds compile DBs for FFmpeg, OpenSSL, ImageMagick, PHP-SRC
│
├── app/                               # Low-level Clang parser wrappers and shared data types
│
├── data/                              # Output directory — one subdirectory per project (e.g. linux/, FFmpeg/)
├── repositories/                      # Cloned source repos (linux, FFmpeg, etc.) with full git history
└── workspace/                         # Temporary working files created during pipeline execution
```

---

## Prerequisites

### Hardware

| Requirement | Minimum |
|---|---|
| OS | Linux x86-64 (Ubuntu 22.04) |
| RAM | 16 GB |
| Disk | 50 GB |

### System packages

```bash
sudo apt-get install -y \
    git \
    build-essential \
    bear \
    openjdk-21-jdk \
    python3-dev \
    curl \
    jq
```

Verify Bear and Java:
```bash
bear --version
java -version    # must be 21+
```

---

## Environment Setup

The pipeline uses a **conda environment with Python 3.7** (`tempovic`), defined in `environment.yml` at the repo root.

### Key packages installed by environment.yml

| Category | Packages |
|---|---|
| **Python** | `python=3.7.16` |
| **Deep Learning** | `pytorch=1.8.0`, `torch-geometric=2.2.0`, `torchvision`, `torchaudio` |
| **Graph / ML** | `networkx=2.7`, `scikit-learn`, `scipy`, `xgboost` |
| **NLP / Transformers** | `transformers=4.30.2`, `tokenizers`, `gensim`, `nltk` |
| **C / Clang parsing** | `clang=12.0.1`, `libclang=18.1.1` |
| **Git analysis** | `pydriller=1.15.2`, `gitpython`, `unidiff` |
| **Code analysis** | `lizard`, `levenshtein`, `rapidfuzz` |
| **General** | `pandas`, `numpy`, `pyyaml`, `tqdm`, `matplotlib` |
| **Java (for Joern)** | `openjdk=21.0.6`, `maven` |

### 1. Install Miniconda (if not already installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 2. Create and activate the environment

```bash
cd /path/to/TempoVIC          # repo root (where environment.yml lives)
conda env create -f environment.yml
conda activate tempovic
```

### 3. Fix Python PATH priority (important)

If your system has a `python` or `python3` binary in `~/bin` that overrides conda, remove or check it:

```bash
ls -la ~/bin/python* 2>/dev/null
```

If those are symlinks to the system Python, remove them so conda's interpreter takes priority:

```bash
rm ~/bin/python ~/bin/python3 ~/bin/python3.12   # adjust as needed
```

Verify the correct Python is active after reactivating:

```bash
conda activate tempovic
which python      # should point to miniconda3/envs/tempovic/bin/python
python --version  # should show Python 3.7.x
```

### 4. Set the libclang environment variable

Add this to your `~/.bashrc`:

```bash
export TEMPOVC_LIBCLANG="$(python -c "import clang, os; print(os.path.join(os.path.dirname(clang.__file__), 'native', 'libclang.so'))")"
```

Then reload:
```bash
source ~/.bashrc
conda activate tempovic
```

---

## Tool Setup

### Bear

Bear intercepts compiler calls during the kernel build to produce `compile_commands.json`, which Clang needs to parse kernel source files.

Install via apt (see Prerequisites). Verify:
```bash
bear --version
```

#### Generate compile_commands.json for Linux

```bash
cd TempoVIC/graph_construction

# Clone the Linux kernel (full history required for V-SZZ git blame)
mkdir -p repositories
git clone https://github.com/torvalds/linux.git repositories/linux

# Run Bear for all kernel builds (multi-arch / multi-version shard build + merge)
python3 bear_compilation/build_linux_kernel_compile_dbs.py
```

Output is written to: `repositories/linux/compile_commands.json`

#### Generate compile_commands.json for other projects

FFmpeg, OpenSSL, ImageMagick, and PHP-SRC use a separate script:

```bash
cd TempoVIC/graph_construction
python3 bear_compilation/build_general_projects_compile_db.py
```

Each `compile_commands.json` is written by Bear in the corresponding repo under `repositories/<Project>/`.

---

### GumTree

GumTree performs AST-level diff between before/after source versions to map patch line numbers accurately.

#### Build from source

```bash
cd TempoVIC/graph_construction/mapping
git clone https://github.com/GumTreeDiff/gumtree.git gumtree
cd gumtree

# If a previous partial build exists, clean it first
rm -rf dist/build/install/gumtree

# Build the shadow distribution
./gradlew :dist:installShadowDist
```

The binary is placed at: `mapping/gumtree/dist/build/install/gumtree/bin/gumtree`

This path is already set in `config.yaml`. No further configuration needed.

---

### Joern

Joern generates Code Property Graphs (CPG) from C source files.

#### Install

```bash
mkdir -p ~/bin/joern
cd ~/bin/joern
curl -L "https://github.com/joernio/joern/releases/latest/download/joern-install.sh" \
    -o joern-install.sh
chmod +x joern-install.sh
./joern-install.sh --install-dir .
```

Add Joern to PATH in `~/.bashrc`:
```bash
export PATH="$HOME/bin/joern/joern-cli:$PATH"
```

Reload and verify:
```bash
source ~/.bashrc
joern --version
```

---

## Configuration

All paths and settings are managed in `graph_construction/config.yaml`. Typed getter functions in `config_loader.py` expose them to the pipeline.

### Machine-specific values

Two values must be set per machine (not committed to the repo):

| Setting | How to set |
|---|---|
| `libclang_path` | Set `TEMPOVC_LIBCLANG` env var (see Environment Setup above) |
| `joern` binary | Must be in `PATH` |

Everything else in `config.yaml` uses paths relative to `graph_construction/` and works once the tools are built.

### Switching projects at runtime

```bash
# Process FFmpeg (no file edits required)
export TEMPOVC_PROJECT=FFmpeg
python extract_source_files.py --mode fix

# Or pass explicitly via --projects
python extract_source_files.py --mode fix --projects FFmpeg OpenSSL
```

### Verify config resolves correctly

```bash
cd graph_construction
python -c "
from config_loader import get_libclang_path, get_gumtree_bin_path, get_project_compile_db
print('libclang  :', get_libclang_path())
print('gumtree   :', get_gumtree_bin_path())
print('compile_db:', get_project_compile_db('linux'))
"
```

All three should print real paths with no errors before proceeding.

---

## Data Preparation

Each test case requires an `info.json` file at: `graph_construction/data/linux/<test_name>/info.json`

### info.json schema

```json
{
    "cve_id": "CVE-YYYY-NNNNN",
    "fix": "<40-character fix commit SHA>",
    "induce": ["<40-character ground truth inducing commit SHA>"]
}
```

### Using the provided dataset

Download the pre-formatted dataset from OSF **[TempoVIC Full Dataset (OSF)](https://osf.io/u5s9v/overview?view_only=be083430853b4c41b56003e862995b81)** and extract it into the `graph_construction/` directory.

The structure for each test case will look like this:
```
data/linux/<test_name>/
├── info.json
├── commits.json                         (optional)
├── <fix_commit_hash>/
│   ├── graph.json
│   └── graph_vszz_full_history.json
└── <history_commit_hash>/
    ├── graph.json
    └── graph_vszz_full_history.json
```

---

## Pipeline Steps

All scripts are run from `graph_construction/` with the conda environment active.

---

### Step 1 — Extract Source Files

Checks out the before/after/fixing C source files for each fixing commit.

```bash
# Linux only
python extract_source_files.py --mode fix --projects linux

# All configured projects
python extract_source_files.py --mode fix

# Specific subset
python extract_source_files.py --mode fix --projects FFmpeg OpenSSL
```

**Output per test case:**
```
data/linux/<test_name>/<fix_sha>/before/    ← C files before the fix
data/linux/<test_name>/<fix_sha>/after/     ← C files after the fix
data/linux/<test_name>/<fix_sha>/fixing/    ← the .patch file
```

---

### Step 2 — Generate Joern CPG

Runs Joern to generate CPG dot files for all `before/`, `after/` source directories.

```bash
python generate_joern.py --projects linux
```

Joern runs **once** for all targets. The script writes `mapping/joern_targets.txt` (one `srcDir;outDir` pair per line) and then invokes `genJoern.sc` which processes them all.

**Output per source directory:**
```
data/linux/<test_name>/<fix_sha>/before/joern/   ← dot files per method
data/linux/<test_name>/<fix_sha>/after/joern/
```

---

### Step 3 — Build Graphs (Fix Commit)

Parses the CPG dot files and constructs heterogeneous `graph.json` for each fixing commit.

```bash
python generate_graphs.py --mode fix --projects linux
```

**Output:**
```
data/linux/<test_name>/<fix_sha>/graph.json
```

**Skip conditions** (reported as `[SKIP]`):
- Both `before/joern/` and `after/joern/` are empty → run Step 2 first
- No patch file in `fixing/`
- Assembly-only patch (`.S` files, no `.c` or `.h`)
- Addition-only patch (no deleted lines to trace)

---

### Step 4 — Run V-SZZ

Traces deleted lines in `graph.json` back through git history to identify candidate inducing commits.

```bash
python history_commits_retrieval.py --projects linux
```

Requires the Linux kernel repo to be cloned at `repositories/linux` with full git history.

**Output:**
```
data/linux/<test_name>/<fix_sha>/graph_vszz_full_history.json
```

---

### Step 4b — Create commits.json

Reads `graph_vszz_full_history.json` and collects all history commits into a structured `commits.json` file used by Steps 5–7.

```bash
python create_commits_json.py --projects linux
```

**Output:** `data/linux/<test_name>/commits.json`

**commits.json schema:**
```json
{
    "fix_commit": "abc123...",
    "ground_truth": ["def456..."],
    "vszz_introducer_commits": ["ghi789..."],
    "all_commits_in_history": ["jkl012...", "..."],
    "matches_ground_truth": true,
    "graph_type": "vszz_full_history",
    "stats": {
        "num_introducers": 3,
        "num_history_commits": 47,
        "nodes_with_history": 12
    }
}
```

---

### Step 5 — Extract History Source Files

Checks out source files for every commit listed in `commits.json`.

```bash
python extract_source_files.py --mode history --projects linux
```

**Output per history commit:**
```
data/linux/<test_name>/<history_sha>/before/
data/linux/<test_name>/<history_sha>/after/
data/linux/<test_name>/<history_sha>/fixing/
```

---

### Step 6 — Generate Joern CPG (History)

Runs Joern on all history commit source directories.

```bash
python generate_joern.py --projects linux
```

This is the same command as Step 2. It picks up any new directories not yet processed.

---

### Step 7 — Build Graphs (History Commits)

Builds `graph.json` for every history commit.

```bash
python generate_graphs.py --mode history --projects linux
```

---

## Running a Specific Project

Every script accepts `--projects` to restrict processing:

```bash
# Single project
python extract_source_files.py --mode fix --projects FFmpeg

# Multiple projects
python generate_graphs.py --mode fix --projects FFmpeg OpenSSL PHP-SRC

# All configured projects (default when --projects is omitted)
python generate_graphs.py --mode fix
```

Configured projects: `linux`, `FFmpeg`, `ImageMagick`, `OpenSSL`, `PHP-SRC`.  
To add a project, add an entry to the `projects:` block in `config.yaml` — no script changes required.

---

## Output Structure

The directory layout after running the complete pipeline:

```
graph_construction/
├── data/
│   └── linux/
│       └── test42/
│           ├── info.json                       # CVE metadata: fix SHA + ground truth
│           ├── commits.json                    # All history commits discovered (Step 4b)
│           ├── <fix_sha>/
│           │   ├── before/                     # Source files before the fix (Step 1)
│           │   ├── after/                      # Source files after the fix (Step 1)
│           │   ├── fixing/                     # Patch file (Step 1)
│           │   ├── graph.json                  # Heterogeneous graph — fix commit (Step 3)
│           │   └── graph_vszz_full_history.json # Full history graph with inducer chains (Step 4)
│           └── <history_sha>/
│               ├── before/
│               ├── after/
│               ├── fixing/
│               ├── graph.json                  # Heterogeneous graph — history commit (Step 7)
│               └── graph_vszz_full_history.json
├── repositories/
│   └── linux/
│       ├── compile_commands.json               # Bear output (Step 0)
│       └── ...                                 # Full kernel source + git history
└── mapping/
    └── joern_targets.txt                       # Generated list of Joern source→output pairs
```

> The structure is identical for each supported project (`linux`, `FFmpeg`, `ImageMagick`, `OpenSSL`, `PHP-SRC`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `libclang not found` | `TEMPOVC_LIBCLANG` not set | See [Environment Setup §4](#4-set-the-libclang-environment-variable) |
| `joern: command not found` | Joern not in PATH | Add `~/bin/joern/joern-cli` to PATH |
| `gumtree: No such file` | GumTree not built | Run `./gradlew :dist:installShadowDist` in `mapping/gumtree/` |
| `[SKIP] joern dirs empty` | Step 2 not run / failed | Re-run `generate_joern.py` and check Joern logs |
| Wrong Python version | System Python overrides conda | Remove `~/bin/python*` symlinks; see §3 above |
| Bear produces empty JSON | Kernel not configured | Run `make defconfig` before Bear-wrapped build |