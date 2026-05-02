# SZZ Baselines for TempoVIC

This directory contains the SZZ baseline implementations used for comparison against the TempoVIC approach. The codebase has been streamlined for C/C++ projects and includes scripts for both executing the baselines and evaluating their performance against our curated ground-truth vulnerability dataset.

> **Acknowledgments:** This codebase was originally adapted from [juzizi44/LLM_SZZ](https://github.com/juzizi44/LLM_SZZ) and has been refined for C/C++ datasets.
## Supported Baselines

- **B-SZZ** (`b_szz.py`): The standard Base SZZ algorithm.
- **AG-SZZ** (`ag_szz.py`): Annotation-Graph SZZ, an extension of B-SZZ with size thresholds and structure awareness.
- **MA-SZZ** (`ma_szz.py`): Meta-Change-Aware SZZ.
- **V-SZZ** (`v_szz.py`): Vulnerability-SZZ. Traces deleted lines iteratively through history using Levenshtein distance matching. 
  *(Note: A more advanced graph-aware variant `MySZZAST` is used directly by the TempoVIC graph construction pipeline).*

## Prerequisites

### 1. Data Download
You must place the ground-truth CVE dataset into the `data/` directory.

The dataset file `verified_cve_with_versions_C.json` can be downloaded from here: [TempoVIC OSF Repository](https://osf.io/x3jsn/overview?view_only=ca42e2bb881142328472d668e0f20764)

Place the downloaded file exactly here:
```
baselines/SZZ/data/verified_cve_with_versions_C.json
```

### 2. Repositories
The baselines share the cloned git repositories with the graph construction pipeline. Ensure that the repositories (e.g., `linux`, `FFmpeg`, `OpenSSL`) are cloned and available at:
```
../../graph_construction/repositories/
```

## Usage

### 1. Running the Baselines

Use `main.py` to run the SZZ baselines. By default, it processes the C dataset. 
The `--time` parameter acts as an output folder name for the given run (e.g., `run1`).

```bash
# Run Base-SZZ (B-SZZ)
python3 main.py --method b --language C --time run1

# Run V-SZZ
python3 main.py --method v --language C --time run1

# Run Annotation-Graph SZZ
python3 main.py --method ag --language C --time run1

# Run Meta-Change-Aware SZZ
python3 main.py --method ma --language C --time run1
```

The output JSON files tracking the predicted inducing commits will be generated inside `results/<method>-szz/C/<time>/`.

### 2. Evaluating the Baselines

Once the predictions have been generated, you can evaluate them against the ground truth using `evaluate.py`. 

The evaluation script uses a 12-character SHA prefix match and calculates **Precision**, **Recall**, and **F1-Score** globally across the entire dataset. It also provides a per-project breakdown.

```bash
# Evaluate Base-SZZ (B-SZZ)
python3 evaluate.py --method b --language C --time run1

# Evaluate V-SZZ
python3 evaluate.py --method v --language C --time run1

# Evaluate Annotation-Graph SZZ
python3 evaluate.py --method ag --language C --time run1

# Evaluate Meta-Change-Aware SZZ
python3 evaluate.py --method ma --language C --time run1
```

## Code Structure

- `main.py`: Entry point for running the baselines.
- `evaluate.py`: Calculates unified global metrics against the ground truth.
- `setting.py`: Configuration constants and shared relative path resolutions.
- `data_loader.py`: Handles loading and mapping from the `verified_cve_with_versions_C.json` dataset.
- `log_generation.py`: Contains Git operations for logging and diff analysis.
- `szz/`: Contains the specific SZZ baseline algorithm implementations.
- `szz/core/`: Contains the shared `AbstractSZZ` logic and the C/C++ `srcML`-based comment parser.
