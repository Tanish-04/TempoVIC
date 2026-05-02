# NeuralSZZ Baseline for TempoVIC

This directory contains the NeuralSZZ baseline implementation — a graph neural network approach that uses a Heterogeneous Attention Network (HAN) combined with RankNet to rank rootcause lines to identify vulnerability-inducing commit candidates.

> **Acknowledgments:** This codebase was originally adapted from the [NeuralSZZ paper](https://baolingfeng.github.io/papers/ASE2023.pdf)
## Overview

NeuralSZZ works in two stages:
1. **B-SZZ Label Generation** — Runs traditional B-SZZ on each deletion line in the fixing commit graph to identify candidate inducing commits (labels for training).
2. **Graph-based Ranking** — Builds mini subgraphs around each deletion node, trains a HAN + RankNet model to rank which deletion lines are most likely to point to the true vulnerability-inducing commit.

## Prerequisites

### Data
The pipeline requires completed graph construction output from `graph_construction/`. Specifically, each test case needs:
- `info.json` — Contains `fix` (fixing commit SHA) and `induce` (ground-truth inducing commits)
- `<fix_sha>/graph.json` — The code change graph from graph construction

The dataset file can be downloaded from: [TempoVIC OSF Repository](https://osf.io/x3jsn/overview?view_only=ca42e2bb881142328472d668e0f20764)

### Dependencies
- PyTorch, PyTorch Geometric
- Transformers (for UniXcoder: `microsoft/unixcoder-base-nine`)
- scikit-learn, Levenshtein

### Repositories
Git repositories must be cloned at `graph_construction/repositories/` (shared with the graph construction pipeline).

## Pipeline

### Step 1: Run B-SZZ Label Generation

Runs B-SZZ on each deletion node in `graph.json` to find candidate inducing commits. Writes `graph_bszz.json` per test case.

```bash
python run_bszz.py --projects linux
python run_bszz.py --projects linux FFmpeg
```

### Step 2: Generate Mini Graphs

Extracts mini subgraphs (DFS neighborhoods) around each deletion node, tokenizes code with UniXcoder, and saves to `data/miniGraphs_bszz.json`.

```bash
python genMiniGraphs.py --projects linux
python genMiniGraphs.py --projects linux FFmpeg
```

### Step 3: Train

Trains the HAN + RankNet model with a 70/15/15 train/val/test split.

```bash
python train.py --seed 456
python train.py --seed 456 --device cuda:1
```

Outputs:
- `checkpoints_baseline_seed456/` — Best model weights (`han_best.pt`, `ranknet_best.pt`)
- `results_baseline_seed456/` — Training logs and final test metrics

### Step 4: Evaluate with SZZ Pipeline

After training, loads the trained model, ranks deletion lines, runs B-SZZ / V-SZZ / AG-SZZ on the top-k ranked lines, and computes Precision / Recall / F1.

```bash
# Run all SZZ variants on top-3 ranked lines
python eval_szz_top_ranked.py --seed 456 --top-k 3

# Run only B-SZZ on test split
python eval_szz_top_ranked.py --seed 456 --bszz-only --eval-split test

# Include AG-SZZ (slower)
python eval_szz_top_ranked.py --seed 456 --top-k 3 --agszz
```

## Code Structure

| File | Description |
|------|-------------|
| `config.py` | Shared path configuration (GRAPH_DATA_DIR, REPOS_DIR, SZZ_DIR, etc.) |
| `run_bszz.py` | Step 1 — B-SZZ label generation on graph.json |
| `genMiniGraphs.py` | Step 2 — Mini subgraph extraction + tokenization |
| `train.py` | Step 3 — HAN + RankNet training with early stopping |
| `eval_szz_top_ranked.py` | Step 4 — Post-training SZZ pipeline evaluation |
| `model.py` | HAN (UniXcoder + HANConv) and RankNet model definitions |
| `genPyG.py` | Converts mini graph JSON to PyTorch Geometric HeteroData |
| `genPairs.py` | Generates pairwise training samples for RankNet |
| `genBatch.py` | Batches pairs for GPU training |
| `eval.py` | Scoring, ranking, and metric functions used during training |
