# TempoVIC: Temporal Graph-based Vulnerability-Inducing Commit Identification

> **TempoVIC** is a two-phase, graph neural network approach for automatically identifying the commit that introduced a vulnerability in C/C++ projects, given the commit that patched it.

This repository contains the complete code implementation accompanying the paper, including graph construction, the main training pipeline with ablation variants, and all baseline methods.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Contributions](#key-contributions)
3. [Repository Structure](#repository-structure)
4. [Dataset](#dataset)
5. [Environment Setup](#environment-setup)
6. [End-to-End Pipeline](#end-to-end-pipeline)
   - [Stage 1: Graph Construction](#stage-1-graph-construction)
   - [Stage 2: Graph Pre-Building](#stage-2-graph-pre-building)
   - [Stage 3: Training](#stage-3-training)
7. [Baselines](#baselines)
8. [Model Variants & Ablations](#model-variants--ablations)
9. [Configuration Reference](#configuration-reference)
10. [Citation](#citation)
11. [Acknowledgments](#acknowledgments)

---

## Overview

Traditional SZZ-based approaches trace vulnerability-inducing commits (VICs) by performing a line-level `git blame` from the fixing commit. While effective for simple cases, these approaches struggle with complex vulnerability chains spanning many commits over long time horizons.

**TempoVIC** improves VIC identification by encoding both the structural (CFG, DFG) and **temporal** relationships between code changes into a heterogeneous graph, and then learning to rank candidate commits using a two-phase architecture:

```
Fixing Commit
      │
      ▼
┌────────────────────────────────────────-┐
│  Phase 1: Root-Cause Line Identification│
│  ─────────────────────────────────────  │
│  GAT encoder over heterogeneous         │
│  code graph (CFG + DFG + temporal)      │
│  + RankNet pairwise scoring head        │
│  → Surfaces the deletion lines most     │
│    likely to be the root cause          │
└────────────────┬────────────────────────┘
                 │ top-k deletion line embeddings
                 ▼
┌────────────────────────────────────────┐
│  Phase 2: VIC Ranking                  │
│  ───────────────────────────────────── │
│  Commit Transformer + dual-query       │
│  cross-attention                       │
│  → Ranks historical commits to find    │
│    the vulnerability-inducing one      │
└────────────────────────────────────────┘
```

The temporal edges in the graph encode commit ordering and time-distance between code changes, allowing the model to reason about *when* a piece of code was introduced relative to other changes — a signal that purely structural approaches miss.

---

## Key Contributions

- **Temporal Code Graphs:** A novel graph representation that augments CFG/DFG edges with temporal edges encoding the commit history of each code line, enabling the model to capture vulnerability introduction patterns across time.
- **Two-Phase Architecture:** A cascaded pipeline where Phase 1 narrows the search space to the most suspicious deletion lines, and Phase 2 ranks full commits.
- **Comprehensive Evaluation:** Evaluation on 627 real-world Linux kernel CVEs with ground-truth inducing commits, compared against four SZZ baselines and NeuralSZZ.

---

## Repository Structure

```text
TempoVIC/
├── README.md                   ← Main Project Readme
├── environment.yml             ← Conda environment specification
├── requirements.txt            ← pip dependencies
│
├── graph_construction/         ← Stage 1: Build heterogeneous code graphs from CVE commits
│   ├── README.md               ← Detailed 7-step pipeline documentation
│   ├── config.yaml             ← Graph construction configuration (projects, tools, paths)
│   ├── extract_source_files.py ← Extract before/after source files from git
│   ├── generate_joern.py       ← Run Joern CPG analysis
│   ├── generate_graphs.py      ← Build heterogeneous graphs (CFG + DFG + line mapping)
│   ├── history_commits_retrieval.py ← V-SZZ based history commit tracing
│   ├── create_commits_json.py  ← Generate commits.json for each test case
│   ├── gen_finalgraph/         ← Final graph assembly with temporal edges
│   ├── parse_patch/            ← Unified diff parsing
│   ├── parse_joerndot/         ← Joern .dot file parsing
│   ├── mapping/                ← Line mapping between file versions
│   ├── data/                   ← Output: per-project test case directories
│   └── repositories/           ← Cloned git repositories (linux, FFmpeg, etc.)
│
├── training_pipeline/          ← Stage 2-3: Pre-build graphs, train models
│   ├── README.md               ← Training pipeline documentation
│   ├── training_config.yaml    ← Central configuration (hyper-parameters, paths, model)
│   ├── config_utils.py         ← YAML config loader with path resolution
│   ├── build_graphs.py         ← Pre-compute PyG-compatible temporal/non-temporal graphs
│   ├── main.py                 ← Main training entry point (Phase 1 → Phase 2)
│   ├── data_processing/        ← Dataset classes, constants, mini-graph extraction
│   ├── models/                 ← Neural network architectures
│   │   ├── phase1_model.py     ← GAT + RankNet for root-cause line identification
│   │   ├── phase2_model.py     ← Commit Transformer with dual-query attention
│   │   ├── shared_encoder.py   ← GAT encoder with UniXcoder embeddings
│   │   └── variants/           ← Ablation architectures (DeepSets, Transformer)
│   └── training/               ← Training loops, losses, evaluation, utilities
│
└── baselines/                  ← Baseline implementations for comparison
    ├── SZZ/                    ← Traditional SZZ family (B-SZZ, V-SZZ, AG-SZZ, MA-SZZ)
    │   └── README.md
    └── neuralszz/              ← NeuralSZZ baseline (HAN + RankNet)
        └── README.md
```

> Each subdirectory contains its own `README.md` with detailed instructions for that module.

---

## Dataset

The dataset consists of **627 verified Linux kernel CVEs** with ground-truth vulnerability-inducing commits.

Each test case contains:

| File | Description |
|------|-------------|
| `info.json` | Fixing commit SHA, inducing commit SHA(s), CVE ID |
| `<fix_sha>/graph.json` | Heterogeneous code change graph |
| `commits.json` | Historical commit chain for temporal graph construction |

**Download:** The dataset is available on the [TempoVIC OSF Repository](https://osf.io/x3jsn/overview?view_only=ca42e2bb881142328472d668e0f20764).

### Supported Projects

| Project | Language | Status |
|---------|----------|--------|
| Linux kernel | C | ✅ Primary evaluation (627 CVEs) |
| FFmpeg | C | ✅ Supported |
| OpenSSL | C | ✅ Supported |
| ImageMagick | C | ✅ Supported |
| PHP-SRC | C | ✅ Supported |

---

## Environment Setup

### Prerequisites

| Requirement | Details |
|-------------|---------|
| OS | Linux (tested on Ubuntu 20.04+) |
| GPU | CUDA-compatible, ≥16 GB VRAM recommended |
| Package Manager | Conda (Miniconda or Anaconda) |
| Java | Java 21 (for GumTree) |
| CPG Generator | Joern |
| Build Tool | Bear (for `compile_commands.json`) |

### Installation

```bash
# Clone the repository
git clone https://github.com/<org>/TempoVIC.git
cd TempoVIC

# Create the conda environment
conda env create -f environment.yml
conda activate tempovic

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

### Tool Setup

For graph construction, the following additional tools must be installed:

- **[Joern](https://joern.io/)** — Code Property Graph generator for C/C++
- **[GumTree](https://github.com/GumTreeDiff/gumtree)** — AST differencing tool
- **[Bear](https://github.com/rizsotto/Bear)** — Compilation database generator for `compile_commands.json`

> See [`graph_construction/README.md`](graph_construction/README.md) for detailed tool installation instructions.

---

## End-to-End Pipeline

The complete pipeline from raw CVE data to trained model runs in three stages:

```
Stage 1                    Stage 2                  Stage 3
Graph Construction   →   Graph Pre-Building   →    Training
(run once)               (PyG conversion)           (Phase 1 → Phase 2)
```

### Stage 1: Graph Construction

Transforms each CVE's fixing commit into a heterogeneous code graph with CFG, DFG, and temporal edges. **This is the most time-intensive stage and only needs to be run once.**

```bash
cd graph_construction

# Step 1: Extract source files from git
python extract_source_files.py --mode fix --projects linux

# Step 2: Generate Joern CPG
python generate_joern.py --mode fix --projects linux

# Step 3: Build fixing commit graphs
python generate_graphs.py --mode fix --projects linux

# Step 4: Trace history commits via V-SZZ
python history_commits_retrieval.py --projects linux

# Steps 5–7: Extract history source files, generate CPGs, build history graphs
python extract_source_files.py --history
python generate_joern.py --history
python generate_graphs.py --history
```

> **Note:** See [`graph_construction/README.md`](graph_construction/README.md) for the full 7-step pipeline with detailed instructions and troubleshooting.

---

### Stage 2: Graph Pre-Building

Converts raw JSON graphs into PyG-compatible subgraphs for efficient training.

```bash
cd training_pipeline

# Build temporal graphs (main model)
python build_graphs.py --mode full_graph

# Build non-temporal graphs (ablation variants)
python build_graphs.py --mode no_temporal

# Build both
python build_graphs.py --mode all
```

---

### Stage 3: Training

The two-phase training runs sequentially: Phase 1 trains the root-cause line ranker, then Phase 2 uses those embeddings to train the commit ranker.

```bash
cd training_pipeline

# Run complete training pipeline (Phase 1 → Phase 2)
python main.py

# Skip Phase 1 if a checkpoint already exists
python main.py --skip-phase1

# Train an ablation variant
python main.py --encoder-type deepsets --graph-mode no_temporal
```

All hyper-parameters are configured via [`training_pipeline/training_config.yaml`](training_pipeline/training_config.yaml).
See [`training_pipeline/README.md`](training_pipeline/README.md) for detailed instructions on Stage 2 and 3

---

## Baselines

### SZZ Family

Traditional line-tracing approaches that use `git blame` to trace deletion lines to their introducing commit.

| Method | Description |
|--------|-------------|
| **B-SZZ** | Basic SZZ — direct `git blame` on modified lines |
| **V-SZZ** | Vulnerability-aware SZZ with Levenshtein distance matching |
| **AG-SZZ** | Annotation-Graph SZZ with size thresholds and structure awareness |
| **MA-SZZ** | Meta-Change-Aware SZZ |

```bash
cd baselines/SZZ
python main.py --method b --language C --time run1   # B-SZZ
python main.py --method v --language C --time run1   # V-SZZ
python evaluate.py --language C --time run1          # Evaluate
```

> See [`baselines/SZZ/README.md`](baselines/SZZ/README.md) for full details.

---

### NeuralSZZ

A graph neural network baseline using HAN (Heterogeneous Attention Network) + RankNet for ranking deletion lines.

```bash
cd baselines/neuralszz

# Step 1: B-SZZ label generation
python run_bszz.py --projects linux

# Step 2: Generate mini graphs
python genMiniGraphs.py --projects linux

# Step 3: Train
python train.py --seed 456

# Step 4: Evaluate with SZZ pipeline
python eval_szz_top_ranked.py --seed 456 --top-k 3
```

> See [`baselines/neuralszz/README.md`](baselines/neuralszz/README.md) for full details.

---

## Model Variants & Ablations

Three architectural variants are supported to study the contribution of temporal edges and graph structure:

| Variant | `encoder_type` | `graph_mode` | Temporal Edges | Description |
|---------|---------------|-------------|:--------------:|-------------|
| **TempoVIC (Main)** | `gat` | `full_graph` | ✅ | GAT encoder with temporal + structural edges |
| **SectionTransformer** | `section_transformer` | `no_temporal` | ❌ | Transformer encoder over commit sections |
| **DeepSets** | `deepsets` | `no_temporal` | ❌ | Order-agnostic pooling architecture |

The ablation variants isolate the contribution of temporal edges by operating on `no_temporal` graphs, making them directly comparable to the main model.

---

## Configuration Reference

### Graph Construction (`graph_construction/config.yaml`)

| Key | Description |
|-----|-------------|
| `projects.<name>.repo_name` | Git repository name |
| `pipeline.repos_dir` | Path to cloned repositories |
| `vszz.pyszz_path` | Path to SZZ implementations |

### Training (`training_pipeline/training_config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `paths.data_root` | `../graph_construction/data/linux` | Path to constructed graphs |
| `paths.graph_mode` | `full_graph` | Graph variant to load |
| `model.encoder_type` | `gat` | Architecture: `gat`, `deepsets`, `section_transformer` |
| `phase1.epochs` | `20` | Phase 1 training epochs |
| `phase1.bert_freeze_bottom_layers` | `8` | UniXcoder layers to freeze |
| `phase2.epochs` | `100` | Phase 2 training epochs |
| `phase2.top_k_lines` | `4` | Number of top Phase 1 lines used in Phase 2 |
| `phase2.num_commit_transformer_layers` | `3` | Depth of commit Transformer |
| `phase2.use_dual_query` | `True` | Enable dual-query cross-attention |
| `phase2.use_temporal_pe` | `True` | Enable temporal positional encodings |
| `defaults.seed` | `456` | Random seed for reproducibility |


---

## Acknowledgments

- **SZZ Baselines:** Adapted from [juzizi44/LLM_SZZ](https://github.com/juzizi44/LLM_SZZ), refined for C/C++ datasets.
- **NeuralSZZ:** Adapted from the [NeuralSZZ paper](https://baolingfeng.github.io/papers/ASE2023.pdf).
- **Code Embeddings:** [UniXcoder](https://github.com/microsoft/CodeBERT/tree/master/UniXcoder) (`microsoft/unixcoder-base-nine`).
- **CPG Generation:** [Joern](https://joern.io/) for Code Property Graph analysis.
- **AST Differencing:** [GumTree](https://github.com/GumTreeDiff/gumtree) for tree-based code differencing.