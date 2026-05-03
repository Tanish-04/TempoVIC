"""
Evaluate Phase 2 commit ranking at @1, @2, @3.

Loads trained Phase 1 + Phase 2 models and evaluates on the held-out test set.
All configuration is read from training_config.yaml via ConfigManager.

Usage:
    python run_evaluation.py                          # main GAT model
    python run_evaluation.py --variant section_transformer
    python run_evaluation.py --variant deepsets

Metric (per case at @k):
    preds = deduplicated top-k ranked commits
    hits  = |preds ∩ GT|
    Precision = Σhits / Σ|preds|
    Recall    = Σhits / Σ|GT|
    F1        = harmonic mean
"""

import argparse
import json
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from config_utils import ConfigManager
from data_processing.dataset import DeletionLineDataset
from models.shared_encoder import UnixcoderEmbedder
from training.embedding_cache import score_deletion_lines, build_phase2_items
from training.evaluation import load_true_commit_map, evaluate_global
from training.utils import build_phase1_model, build_phase2_model, setup_device


# Variant-specific overrides (checkpoint dirs and graph modes for ablations)
VARIANT_OVERRIDES = {
    "gat": {
        "encoder_type":  "gat",
        "graph_mode":    None,       # use config default
        "save_dir":      None,       # use config default
        "p2_checkpoint": "phase2_best.pt",
    },
    "section_transformer": {
        "encoder_type":  "section_transformer",
        "graph_mode":    "no_temporal",
        "save_dir":      "checkpoints_ablation_no_temporal",
        "p2_checkpoint": "phase2_best.pt",
    },
    "deepsets": {
        "encoder_type":  "deepsets",
        "graph_mode":    "no_temporal",
        "save_dir":      "checkpoints_ablation_deepsets",
        "p2_checkpoint": "phase2_best.pt",
    },
}


def load_config(variant_name: str) -> dict:
    """Load training_config.yaml and apply variant overrides."""
    cm = ConfigManager()
    cfg = cm.raw

    variant = VARIANT_OVERRIDES[variant_name]

    # Apply variant overrides (None = keep config default)
    cfg["model"]["encoder_type"] = variant["encoder_type"]

    if variant["graph_mode"] is not None:
        cfg["paths"]["graph_mode"] = variant["graph_mode"]

    if variant["save_dir"] is not None:
        save_dir = variant["save_dir"]
        if not Path(save_dir).is_absolute():
            save_dir = str(Path(__file__).resolve().parent / save_dir)
        cfg["paths"]["save_dir"] = save_dir

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 2 commit ranking at @1, @2, @3"
    )
    parser.add_argument(
        "--variant",
        choices=["gat", "section_transformer", "deepsets"],
        default="gat",
        help="Model variant to evaluate (default: gat)",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.variant)
    variant = VARIANT_OVERRIDES[args.variant]

    data_root    = cfg["paths"]["data_root"]
    save_dir     = cfg["paths"]["save_dir"]
    graph_mode   = cfg["paths"]["graph_mode"]
    prebuilt_dir = cfg["paths"]["prebuilt_dir"]
    test_cases_file = cfg["paths"]["test_cases_file"]
    seed         = cfg["defaults"]["seed"]
    gpu_id       = cfg["defaults"]["gpu_id"]
    top_k_lines  = cfg["phase2"].get("top_k_lines", 4)
    max_nodes    = cfg["defaults"].get("max_nodes_per_batch", 4096)

    p2_checkpoint = variant["p2_checkpoint"]


    # Setup device
    device = setup_device(gpu_id)
    embedder = UnixcoderEmbedder(tokenizer_only=True)

    # Load test cases and split
    with open(Path(data_root) / test_cases_file) as f:
        all_cases = json.load(f)

    _, temp = train_test_split(all_cases, test_size=0.30, random_state=seed)
    _, test_cases = train_test_split(temp, test_size=0.50, random_state=seed)


    # Load dataset
    dataset = DeletionLineDataset(
        data_path=data_root,
        test_cases=all_cases,
        embedder=embedder,
        prebuilt_dir=prebuilt_dir,
        graph_mode=graph_mode,
    )

    # Phase 1: Load model and score deletion lines
    p1_model = build_phase1_model(cfg, device)
    p1_state = torch.load(
        Path(save_dir) / "phase1_best.pt", map_location="cpu"
    )["model_state_dict"]

        
    # The saved checkpoint was produced by an older codebase that named the 
    # unixcoder model bert_model inside SharedEncoder. The refactored
    # code renamed it to unix_model (and related flags to include_unix,
    # unix_chunk) to reflect the switch from CodeBERT to UniXcoder.
    # Without this remapping, load_state_dict(strict=False) would silently skip
    # all unixcoder weights, leaving the encoder randomly initialized. That's why
    # these lines are needed to remap the keys before loading the state dict.
    
    # remove this if you are not using a provided checkpoint.
    p1_state = {
    k.replace("encoder.bert_model.", "encoder.unix_model.")
     .replace("encoder.include_bert", "encoder.include_unix")
     .replace("encoder.bert_chunk", "encoder.unix_chunk"): v
    for k, v in p1_state.items()
}

    p1_model.load_state_dict(p1_state, strict=True)
    p1_model.encoder.eval()
    for param in p1_model.encoder.parameters():
        param.requires_grad = False

    scored = score_deletion_lines(
        p1_model, dataset, all_cases, device,
        max_nodes=max_nodes,
        top_k=top_k_lines,
    )

    p2_items_by_k = build_phase2_items(
        scored, all_cases,
        graph_mode=graph_mode,
        top_k_lines=top_k_lines,
    )
    p2_items = p2_items_by_k[top_k_lines]

    # --- DIAGNOSTIC: Compare with old script stats ---
    for k in range(1, top_k_lines + 1):
        items = p2_items_by_k[k]
        n_valid = sum(1 for i in items if i["valid"])
        n_correct = sum(1 for i in items if i["valid"] and i["is_correct_deletion_line"])
        n_wrong = sum(1 for i in items if i["valid"] and not i["is_correct_deletion_line"])
        print(f"  [top_k={k}] Phase 2 items: {n_valid}/{len(all_cases)} valid")
        print(f"  [top_k={k}] Correct deletion lines: {n_correct} | Wrong: {n_wrong}")

    # Phase 2: Load model
    p2_model = build_phase2_model(cfg, device)
    p2_state = torch.load(
        Path(save_dir) / p2_checkpoint, map_location="cpu"
    )["model_state_dict"]
    p2_model.load_state_dict(p2_state, strict=False)
    p2_model.eval()

    # Build lookups
    item_map = {
        item["test_name"]: item
        for item in p2_items
        if item.get("valid", False)
    }
    true_cid_map = load_true_commit_map(all_cases, data_root)

    results = evaluate_global(
        cases=test_cases,
        item_map=item_map,
        true_cid_map=true_cid_map,
        p2_model=p2_model,
        device=device,
        data_root=Path(data_root),
    )

    for k in (1, 2, 3):
        m = results[k]
        print(f'    @{k}: P={m["precision"]:.4f}  R={m["recall"]:.4f}  F1={m["f1"]:.4f}'
              f'  hits={m["total_hits"]}  ident={m["total_identified"]}  gt={m["total_gt"]}  N={m["n_evaluated"]}')


if __name__ == "__main__":
    main()