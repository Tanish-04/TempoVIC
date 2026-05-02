import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config_utils import ConfigManager
CONFIG = ConfigManager().raw
from data_processing.dataset import (
    DeletionLineDataset,
    CommitRankingDataset,
)
from models.shared_encoder import UnixcoderEmbedder

from training.embedding_cache import score_deletion_lines, build_phase2_items

from training.phase1_trainer import train_phase1_fold
from training.phase2_trainer import train_phase2_fold
from training.utils import build_phase1_model, build_phase2_model, set_seed, setup_device
from training.evaluation import evaluate_topk_metrics, evaluate_global, load_true_commit_map



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified two-phase training: "
                    "Deletion Line Ranking → Commit Ranking"
    )
    # Phase 1
    p.add_argument("--phase1-epochs",type=int)
    p.add_argument("--phase1-lr",type=float)
    p.add_argument("--phase1-bert-lr",type=float)
    p.add_argument("--phase1-rest-lr",type=float)
    p.add_argument("--phase1-bert-freeze-bottom-layers", type=int)
    p.add_argument("--max-graphs-per-batch", type=int)
    p.add_argument("--phase1-patience",    type=int)

    # Phase 2 (all optional; unset → keep values from config.yaml)
    p.add_argument("--phase2-epochs", type=int, default=None)
    p.add_argument("--phase2-lr", type=float, default=None)
    p.add_argument("--phase2-weight-decay", type=float, default=None)
    p.add_argument("--phase2-batch-size", type=int, default=None)
    p.add_argument("--phase2-patience", type=int, default=None)
    p.add_argument("--phase2-dropout", type=float, default=None)
    p.add_argument("--phase2-gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--phase2-temperature", type=float, default=None)
    p.add_argument("--phase2-label-smoothing", type=float, default=None)
    p.add_argument("--phase2-top-k-lines", type=int, default=None)
    p.add_argument("--phase2-focal-gamma", type=float, default=None)
    p.add_argument("--phase2-focal-alpha", type=float, default=None)
    p.add_argument("--phase2-margin", type=float, default=None)
    p.add_argument("--phase2-margin-weight", type=float, default=None)
    p.add_argument("--phase2-hidden-dim", type=int, default=None)
    p.add_argument("--phase2-num-heads", type=int, default=None)
    p.add_argument("--phase2-num-commit-transformer-layers", type=int, default=None)
    p.add_argument("--phase2-max-commits", type=int, default=None)
    p.add_argument("--phase2-max-temporal-dist", type=int, default=None)
    # Python < 3.9 has no BooleanOptionalAction; paired store_const flags (run_full.sh).
    p.set_defaults(phase2_use_dual_query=None, phase2_use_temporal_pe=None)
    p.add_argument(
        "--phase2-use-dual-query",
        dest="phase2_use_dual_query",
        action="store_const",
        const=True,
        help="Enable Phase 2 dual query",
    )
    p.add_argument(
        "--no-phase2-use-dual-query",
        dest="phase2_use_dual_query",
        action="store_const",
        const=False,
        help="Disable Phase 2 dual query",
    )
    p.add_argument(
        "--phase2-use-temporal-pe",
        dest="phase2_use_temporal_pe",
        action="store_const",
        const=True,
        help="Enable phase2.use_temporal_pe in config",
    )
    p.add_argument(
        "--no-phase2-use-temporal-pe",
        dest="phase2_use_temporal_pe",
        action="store_const",
        const=False,
        help="Disable phase2.use_temporal_pe in config",
    )

    # Shared
    p.add_argument("--hidden-dim",    type=int)
    p.add_argument("--num-gt-layers", type=int)
    p.add_argument("--dropout",       type=float)
    p.add_argument("--seed",          type=int)
    p.add_argument("--save-dir",      type=str)
    p.add_argument("--no-stratify",   action="store_true")



    # Model / graph variant 
    p.add_argument("--encoder-type", type=str, help="Model encoder: 'gat', 'section_transformer', 'deepsets'")
    p.add_argument("--graph-mode", type=str, help="Graph dataset to load: 'temporal', 'no_temporal'")
    p.add_argument("--prebuilt-dir", type=str, help="Subdirectory of data_root containing pre-built graphs (e.g. 'temporal_graph', 'no_temporal_graph')")

    # Skip Phase 1
    p.add_argument("--skip-phase1", action="store_true",
                   help="Load existing Phase 1 checkpoint instead of training")
    p.add_argument("--phase1-checkpoint-dir", type=str)
    return p.parse_args()


def _apply_cli(args: argparse.Namespace) -> None:
    """Overwrite CONFIG in-place with any CLI arguments that were provided."""
    # Maps CLI arg name → (section, key_in_section)
    CLI_MAP = {
        "phase1_epochs":                     ("phase1", "epochs"),
        "phase1_lr":                         ("phase1", "lr"),
        "phase1_bert_lr":                    ("phase1", "bert_lr"),
        "phase1_rest_lr":                    ("phase1", "rest_lr"),
        "phase1_bert_freeze_bottom_layers":  ("phase1", "bert_freeze_bottom_layers"),
        "phase1_patience":                   ("phase1", "patience"),
        "max_graphs_per_batch":              ("phase1", "max_graphs_per_batch"),
        "phase2_epochs":                     ("phase2", "epochs"),
        "phase2_lr":                         ("phase2", "lr"),
        "phase2_weight_decay":               ("phase2", "weight_decay"),
        "phase2_batch_size":                 ("phase2", "batch_size"),
        "phase2_patience":                   ("phase2", "patience"),
        "phase2_dropout":                    ("phase2", "dropout"),
        "phase2_gradient_accumulation_steps": ("phase2", "gradient_accumulation_steps"),
        "phase2_temperature":              ("phase2", "temperature"),
        "phase2_label_smoothing":          ("phase2", "label_smoothing"),
        "phase2_top_k_lines":              ("phase2", "top_k_lines"),
        "phase2_focal_gamma":              ("phase2", "focal_gamma"),
        "phase2_focal_alpha":              ("phase2", "focal_alpha"),
        "phase2_margin":                   ("phase2", "margin"),
        "phase2_margin_weight":            ("phase2", "margin_weight"),
        "phase2_hidden_dim":               ("phase2", "hidden_dim"),
        "phase2_num_heads":                ("phase2", "num_heads"),
        "phase2_num_commit_transformer_layers": ("phase2", "num_commit_transformer_layers"),
        "phase2_max_commits":              ("phase2", "max_commits"),
        "phase2_max_temporal_dist":        ("phase2", "max_temporal_dist"),
        "phase2_use_dual_query":           ("phase2", "use_dual_query"),
        "phase2_use_temporal_pe":          ("phase2", "use_temporal_pe"),
        "hidden_dim":                        ("model",  "hidden_dim"),
        "num_gt_layers":                     ("model",  "num_gt_layers"),
        "dropout":                           ("model",  "dropout"),
        "seed":                              ("defaults", "seed"),
        "save_dir":                          ("paths",   "save_dir"),

        # Ablation / variant switches
        "encoder_type":                        ("model", "encoder_type"),
        "graph_mode":                          ("paths", "graph_mode"),
        "prebuilt_dir":                        ("paths", "prebuilt_dir"),
    }
    for cli_key, (section, yaml_key) in CLI_MAP.items():
        val = getattr(args, cli_key, None)
        if val is not None:
            CONFIG.setdefault(section, {})[yaml_key] = val



def _run_phase1(
    train_cases: List[str],
    val_cases: List[str],
    phase1_dataset: DeletionLineDataset,
    p1_ckpt_path: Path,
    skip_p1: bool,
    device: torch.device,
) -> Optional[Dict]:
    """
    Train Phase 1 or load an existing checkpoint.
    """
    if skip_p1:
        if not p1_ckpt_path.exists():
            raise FileNotFoundError(
                f"--skip-phase1 was set but checkpoint not found: {p1_ckpt_path}"
            )
        print(f"\n  Loading Phase 1 checkpoint: {p1_ckpt_path}")
        state = torch.load(p1_ckpt_path, map_location="cpu")["model_state_dict"]
        return {"model_state": state, "loaded_from": str(p1_ckpt_path)}

    result = train_phase1_fold(
        0, train_cases, val_cases, phase1_dataset, CONFIG, device=device
    )
    if result is None:
        return None

    ckpt_path = Path(CONFIG["paths"]["save_dir"]) / "phase1_best.pt"
    torch.save({"model_state_dict": result["model_state"]}, ckpt_path)
    print(f"  ✓ Phase 1 checkpoint saved to {ckpt_path}")
    return result

def diagnose_phase1_accuracy(scored, cases, label, top_k=3):
    correct = sum(
        1 for name in cases
        if name in scored and any(
            mg.rootcause 
            for _, mg, _ in scored[name][:top_k]
        )
    )
    total = sum(1 for name in cases if name in scored)
    print(f"  Phase 1 top-{top_k} correct ({label}): "
          f"{correct}/{total} = {correct/max(total,1)*100:.1f}%")
        
def diagnose_phase1_chain_coverage(scored, cases, label, top_k=3):
    """
    For each test case, count how many of the top-k deletion lines have
    the true VIC in their chain.

    Prints distribution: 0/k, 1/k, 2/k, ..., k/k lines with VIC in chain.
    """
    from collections import Counter
    
    counts = Counter()
    total  = 0
    
    for name in cases:
        if name not in scored:
            continue
        total += 1
        entry = scored[name][:top_k]
        
        # Count how many deletion lines in top-k have true VIC in their chain
        n_with_vic = 0
        for _, mg, _ in entry:
            chain_commits_short = {sha[:12] for sha in mg.tp_to_commit.values() if sha}
            true_vics_short = {sha[:12] for sha in mg.inducing_commits}
            if chain_commits_short & true_vics_short:
                n_with_vic += 1
        
        counts[n_with_vic] += 1
    
    print(f"\n  [{label}] Top-{top_k} chain coverage distribution "
          f"(total={total} cases):")
    for n in range(top_k + 1):
        c = counts.get(n, 0)
        pct = 100 * c / max(total, 1)
        print(f"    {n}/{top_k} deletion lines have true VIC in chain: "
              f"{c:>4} cases ({pct:5.1f}%)")

def diagnose_phase1_pool_growth(scored, cases, label, top_k=3):
    """
    For each test case, measure how the candidate commit pool grows as we
    accumulate top-k deletion lines.
    
    For k=1,2,...,top_k:
        - Total pool size (deduplicated across chains)
        - Raw total (sum of chain lengths, no dedup)
    """
    import statistics
    
    stats_total  = {k: [] for k in range(1, top_k + 1)}
    stats_raw    = {k: [] for k in range(1, top_k + 1)}
    stats_growth = {k: [] for k in range(2, top_k + 1)}  # new commits added
    
    total = 0
    for name in cases:
        if name not in scored:
            continue
        total += 1
        entry = scored[name][:top_k]
        
        accumulated_commits = set()
        for k, (_, mg, _) in enumerate(entry, 1):
            chain_commits = {
                sha[:12] for sha in mg.tp_to_commit.values() if sha
            }
            raw_size = len(chain_commits)
            
            new_commits = chain_commits - accumulated_commits
            accumulated_commits |= chain_commits
            
            stats_total[k].append(len(accumulated_commits))
            stats_raw[k].append(raw_size)
            if k > 1:
                stats_growth[k].append(len(new_commits))
    
    def summary(lst):
        if not lst:
            return "n/a"
        return (f"min={min(lst)}, max={max(lst)}, "
                f"mean={statistics.mean(lst):.2f}, "
                f"median={statistics.median(lst):.1f}")
    
    print(f"\n  [{label}] Pool size growth (total={total} cases):")
    
    print(f"  Per-deletion-line chain length (raw, before pooling):")
    for k in range(1, top_k + 1):
        print(f"    Del{k}: {summary(stats_raw[k])}")
    
    print(f"  Accumulated pool size (deduplicated across chains):")
    for k in range(1, top_k + 1):
        print(f"    Top-{k} total pool: {summary(stats_total[k])}")
    
    print(f"  Marginal new commits added (dedup):")
    for k in range(2, top_k + 1):
        print(f"    Del{k} adds: {summary(stats_growth[k])}")


# All Chains together commits pooling
def _prepare_phase2_data(
    p1_state: Dict,
    phase1_dataset: DeletionLineDataset,
    all_cases: List[str],
    train_cases: List[str],   
    val_cases: List[str],     
    test_cases: List[str],
    device: torch.device,
) -> Tuple[Dict[int, CommitRankingDataset], Dict[str, int]]:

    p1_model = build_phase1_model(CONFIG, device)
    p1_model.load_state_dict(p1_state, strict=False)
    p1_model.encoder.eval()
    for param in p1_model.encoder.parameters():
        param.requires_grad = False

    print(f"\n  Scoring deletion lines for {len(all_cases)} test cases...")
    scored = score_deletion_lines(
        p1_model, phase1_dataset, all_cases, device,
        max_nodes=CONFIG["defaults"]["max_nodes_per_batch"],
        top_k=CONFIG["phase2"]["top_k_lines"],
    )
    print(f"  Top graphs selected: {len(scored)}/{len(all_cases)}")

    # ── Diagnose BEFORE deleting scored ──────────────────────────
    diagnose_phase1_accuracy(scored, train_cases, "train")
    diagnose_phase1_accuracy(scored, val_cases,   "val")
    diagnose_phase1_accuracy(scored, test_cases,  "test")


    # ── NEW: chain coverage diagnostics ──────────────────────────
    top_k_phase1 = CONFIG["phase2"]["top_k_lines"]

    # ── Phase 1 commit ranking @1/@2/@3 (NeuralSZZ-aligned) ──────
    # data_path = CONFIG["paths"]["data_root"]
    # diagnose_phase1_commit_ranking(scored, train_cases, "train", data_path, top_k=top_k_phase1)
    # diagnose_phase1_commit_ranking(scored, val_cases,   "val",   data_path, top_k=top_k_phase1)
    # diagnose_phase1_commit_ranking(scored, test_cases,  "test",  data_path, top_k=top_k_phase1)



    print(f"\n  Phase 1 top-{top_k_phase1} chain coverage (true VIC in chain):")
    diagnose_phase1_chain_coverage(scored, train_cases, "train", top_k=top_k_phase1)
    diagnose_phase1_chain_coverage(scored, val_cases,   "val",   top_k=top_k_phase1)
    diagnose_phase1_chain_coverage(scored, test_cases,  "test",  top_k=top_k_phase1)

    diagnose_phase1_pool_growth(scored, train_cases, "train", top_k=top_k_phase1)
    diagnose_phase1_pool_growth(scored, val_cases,   "val",   top_k=top_k_phase1)
    diagnose_phase1_pool_growth(scored, test_cases,  "test",  top_k=top_k_phase1)

    del p1_model
    gc.collect()

    print("\n  Building Phase 2 embedding items...")
    p2_items_by_k = build_phase2_items(scored, all_cases, graph_mode=CONFIG["paths"]["graph_mode"], top_k_lines=CONFIG["phase2"]["top_k_lines"])

    del scored
    gc.collect()
    torch.cuda.empty_cache()

    # Build one CommitRankingDataset per k
    p2_datasets = {
        k: CommitRankingDataset(items)
        for k, items in p2_items_by_k.items()
    }
    case_to_idx = {name: i for i, name in enumerate(all_cases)}
    return p2_datasets, case_to_idx


# All Commits from all chains pooled together

def diagnose_phase1_commit_ranking(scored, cases, label, data_path, top_k=3):
    """
    Evaluate Phase 1 deletion-line ranking using the same precision/recall/f1
    logic as evaluate_topk_metrics (NeuralSZZ-aligned).
    
    scored: {name: [(score, mg, ...), ...]}  — Phase 1 ranked output
    """

    # Filter to cases that are both requested and scored
    relevant = [c for c in cases if c in scored]

    # Reshape: {name: [mg, mg, ...]} — strip the score/tuple wrapper
    graphs_by_case = {
        name: [mg for _, mg, *_ in scored[name]]
        for name in relevant
    }

    true_cid_map = load_true_commit_map(relevant, data_path)

    print(f"\n  [{label}] Phase 1 commit ranking metrics "
          f"(NeuralSZZ-aligned, {len(relevant)} cases):")

    for k in range(1, top_k + 1):
        m = evaluate_topk_metrics(graphs_by_case, true_cid_map=true_cid_map, k=k)
        print(
            f"    @{k}:  P={m[f'precision@{k}']:.4f}  "
            f"R={m[f'recall@{k}']:.4f}  "
            f"F1={m[f'f1@{k}']:.4f}  "
            f"(TP={m[f'tp@{k}']}, FP={m[f'fp@{k}']}, "
            f"total_gt={m['total_inducing_commits']})"
        )

def _run_phase2(
    train_cases: List[str],
    val_cases: List[str],
    test_cases: List[str],
    p2_datasets: Dict[int, CommitRankingDataset],
    case_to_idx: Dict[str, int],
    device: torch.device,
) -> Dict:
    """
    Train Phase 2 and evaluate on the held-out test set.

    Returns the p2_result dict augmented with ``test_metrics_by_k``.
    """
    train_idx = [case_to_idx[c] for c in train_cases if c in case_to_idx]
    val_idx   = [case_to_idx[c] for c in val_cases   if c in case_to_idx]
    test_idx  = [case_to_idx[c] for c in test_cases  if c in case_to_idx]

    # Train on the configured top_k_lines (from config, e.g. 1)
    train_k  = CONFIG["phase2"]["top_k_lines"]
    train_ds = p2_datasets[train_k]

    # Build item_map and true_cid_map for evaluate_global
    all_eval_cases = train_cases + val_cases + test_cases
    item_map = {
        item["test_name"]: item
        for item in train_ds.items
        if item.get("test_name")
    }
    true_cid_map = load_true_commit_map(all_eval_cases, CONFIG["paths"]["data_root"])
    data_root = Path(CONFIG["paths"]["data_root"])

    p2_result = train_phase2_fold(
        0, train_idx, val_idx, train_ds, CONFIG,
        train_cases=train_cases,
        val_cases=val_cases,
        item_map=item_map,
        true_cid_map=true_cid_map,
        data_root=data_root,
    )

    # ── Test evaluation using evaluate_global at k=1,2,3 ──
    print(f"\n  Evaluating on held-out test set ({len(test_idx)} cases)...")
    test_model = build_phase2_model(CONFIG, device)
    if p2_result.get("best_commit_ranker_state"):
        test_model.load_state_dict(p2_result["best_commit_ranker_state"])

    top_k_max = CONFIG["phase2"]["top_k_lines"]

    # Evaluate on test set for each k by building item_map from the k-th dataset
    test_metrics_by_k = {}
    for k in range(1, top_k_max + 1):
        k_ds = p2_datasets[k]
        k_item_map = {
            item["test_name"]: item
            for item in k_ds.items
            if item.get("test_name")
        }

        print(f"\n  Evaluating test set (top_k_lines={k}, "
              f"{len(test_cases)} cases)...")
        test_results = evaluate_global(
            cases=test_cases,
            item_map=k_item_map,
            true_cid_map=true_cid_map,
            p2_model=test_model,
            device=device,
            data_root=data_root,
        )
        test_metrics_by_k[k] = test_results

        for vk in sorted(test_results.keys()):
            vm = test_results[vk]
            print(
                f"    k={k} @{vk}: P={vm.get('precision', 0):.4f}  "
                f"R={vm.get('recall', 0):.4f}  "
                f"F1={vm.get('f1', 0):.4f}"
            )

    p2_result["test_metrics_by_k"] = test_metrics_by_k

    del test_model
    gc.collect()
    torch.cuda.empty_cache()

    return p2_result



def _run(
    train_cases: List[str],
    val_cases: List[str],
    test_cases: List[str],
    phase1_dataset: DeletionLineDataset,
    p1_ckpt_dir: Path,
    skip_p1: bool,
    device: torch.device,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Execute Phase 1 + Phase 2 for the single train/val/test split."""

    p1_result = _run_phase1(
        train_cases, val_cases,
        phase1_dataset, p1_ckpt_dir / "phase1_best.pt",
        skip_p1, device,
    )
    if p1_result is None:
        return None, None

    all_cases = train_cases + val_cases + test_cases
   

   
    p2_datasets, case_to_idx = _prepare_phase2_data(
        p1_result["model_state"], phase1_dataset, all_cases,
        train_cases, val_cases, test_cases,
        device,
    )
    p2_result = _run_phase2(
        train_cases, val_cases, test_cases,
        p2_datasets, case_to_idx, device,
    )

    torch.save(
        {"model_state_dict": p2_result["best_commit_ranker_state"]},
        Path(CONFIG["paths"]["save_dir"]) / "phase2_best.pt",
    )

    del p2_datasets
    gc.collect()
    torch.cuda.empty_cache()

    return p1_result, p2_result


def _print_and_save(
    p1_result: Dict,
    p2_result: Dict,
    split_info: Dict,
) -> None:
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    if p1_result and "metrics" in p1_result:
        m = p1_result["metrics"]
        print("\nPhase 1 (Deletion Line Ranking)")
        print(f"  P@1:  {m.get('precision@1', 0):.4f}")
        print(f"  R@1:  {m.get('recall@1',    0):.4f}")
        print(f"  F1@1: {m.get('f1@1',        0):.4f}")

    val_m  = p2_result.get("final_metrics", {})

    print("\nPhase 2 — Validation (used for model selection)")
    print(f"  Best epoch : {p2_result.get('best_epoch', 0)}")
    for k in sorted(val_m.keys()):
        km = val_m[k]
        print(
            f"  @{k}: P={km.get('precision', 0):.4f}  "
            f"R={km.get('recall', 0):.4f}  "
            f"F1={km.get('f1', 0):.4f}"
        )

    print("\nPhase 2 — Test (held-out, final numbers)")
    test_metrics_by_k = p2_result.get("test_metrics_by_k", {})
    top_k_max = CONFIG["phase2"]["top_k_lines"]
    for k in range(1, top_k_max + 1):
        k_results = test_metrics_by_k.get(k, {})
        print(f"\n  top_k_lines={k}:")
        for vk in sorted(k_results.keys()):
            vm = k_results[vk]
            print(
                f"    @{vk}: P={vm.get('precision', 0):.4f}  "
                f"R={vm.get('recall', 0):.4f}  "
                f"F1={vm.get('f1', 0):.4f}"
            )

    
    summary = {
        "config":              CONFIG,
        "data_split":          split_info,
        "phase1_metrics":      p1_result.get("metrics", {}),
        "phase2_val_metrics":  val_m,
        "phase2_test_metrics": test_metrics_by_k,
        "best_epoch":          p2_result.get("best_epoch", 0),
    }
    path = Path(CONFIG["paths"]["save_dir"]) / "results_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to {path}")



def main() -> None:
    args = _parse_args()
    _apply_cli(args)

    set_seed(CONFIG["defaults"]["seed"])
    print("Seed check:", torch.rand(3).tolist())

    os.makedirs(CONFIG["paths"]["save_dir"], exist_ok=True)

    p1_ckpt_dir = Path(
        args.phase1_checkpoint_dir
        if args.phase1_checkpoint_dir
        else CONFIG["paths"]["save_dir"]
    )

    device = setup_device(CONFIG["defaults"]["gpu_id"])

    print("=" * 70)
    print("UNIFIED TWO-PHASE TRAINING")
    print(f"  device : {device}")
    print(f"  encoder_type : {CONFIG['model'].get('encoder_type', 'gat')}")
    print(f"  graph_mode   : {CONFIG['paths'].get('graph_mode', 'temporal')}")
    print(f"  save_dir     : {CONFIG['paths']['save_dir']}")
    print("=" * 70)

    with open(Path(CONFIG["paths"]["data_root"]) / CONFIG["paths"]["test_cases_file"]) as f:
        all_cases: List[str] = json.load(f)
    print(f"Total test cases: {len(all_cases)}")

    # Fixed 70 / 15 / 15 split — seeded for reproducibility
    seed = CONFIG["defaults"]["seed"]
    train_cases, temp_cases = train_test_split(
        all_cases, test_size=0.30, random_state=seed, shuffle=True
    )
    val_cases, test_cases = train_test_split(
        temp_cases, test_size=0.50, random_state=seed, shuffle=True
    )

    n = len(all_cases)
    print(f"\nData split (seed={seed}):")
    for label, subset in [("Train", train_cases), ("Val", val_cases), ("Test", test_cases)]:
        print(f"  {label:<6}: {len(subset)} cases ({100*len(subset)/n:.1f}%)")

    split_info = {
        "total": n, "train": len(train_cases),
        "val": len(val_cases), "test": len(test_cases),
        "seed": seed, "strategy": "random_70_15_15",
    }

    print("\nInitialising Unixcoder tokenizer...")
    embedder = UnixcoderEmbedder(tokenizer_only=True)

    print("\nLoading dataset...")
    p1_dataset = DeletionLineDataset(
        data_path=CONFIG["paths"]["data_root"],
        test_cases=all_cases,
        embedder=embedder,
        prebuilt_dir=CONFIG["paths"]["prebuilt_dir"],
        graph_mode=CONFIG["paths"]["graph_mode"],
    )

    
    p1_r, p2_r = _run(
        train_cases, val_cases, test_cases,
        p1_dataset, p1_ckpt_dir, args.skip_phase1, device,
    )
    if p1_r and p2_r:
        _print_and_save(p1_r, p2_r, split_info)

    print(f"\n✓ All outputs saved to: {CONFIG['paths']['save_dir']}")


if __name__ == "__main__":
    main()