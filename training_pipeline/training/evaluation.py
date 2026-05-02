"""
Evaluation metrics for both phases.

Metrics are computed EXACTLY as in NeuralSZZ's train.py (eval_top),
ensuring results are directly comparable to the baseline.

Public API
----------
load_true_commit_map      — load ground-truth inducing commits from info.json
evaluate_topk_metrics     — precision@k / recall@k / f1@k (NeuralSZZ)
evaluate_top1_metrics     — backward-compatible wrapper for k=1
evaluate_ranking          — score every deletion line and compute P/R/F1@1
print_metrics             — pretty-print a metrics dict
compute_summary_statistics — average metrics across folds
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch


def load_true_commit_map(
    test_cases: List[str],
    data_path: str,
) -> Dict[str, set]:
    """
    Load the true bug-inducing commit set for each test case from info.json.

    Returns {test_name: set_of_inducing_commit_shas}.
    """
    result = {}
    base = Path(data_path)
    for name in test_cases:
        path = base / name / "info.json"
        if path.exists():
            with open(path) as f:
                result[name] = set(json.load(f).get("induce", []))
        else:
            result[name] = set()
    return result


def evaluate_topk_metrics(
    test_cases_graphs: Dict[str, List],
    true_cid_map: Optional[Dict[str, set]] = None,
    data_path: Optional[str] = None,
    k: int = 1,
) -> Dict[str, float]:
    """
    Compute precision@k, recall@k, f1@k — aligned with NeuralSZZ eval_top.

    Counting logic (matches BaseLineNeuralSZZ/eval.py::eval_top exactly):
    - For each test case, iterate over the top-k ranked deletion lines.
    - Each deletion line's commits are compared against ground truth.
    - If ANY commit matches ground truth (and hasn't been counted yet) → TP += 1
      (one TP per *line*, not per commit).
    - If NO commit matches but new commits exist → FP += 1.
    - total_t = sum of |ground-truth inducing commits| across all test cases.

    Uses sha[:12] prefix matching consistent with gen_graph.py::rootcause labels.
    """
    if true_cid_map is None:
        if data_path is None:
            raise ValueError(
                "Either true_cid_map or data_path must be provided.")
        true_cid_map = load_true_commit_map(
            list(test_cases_graphs.keys()), data_path)

    tp = fp = total_t = 0

    for test_name, ranked in test_cases_graphs.items():
        if not ranked:
            continue

        gt_set   = true_cid_map.get(test_name, set())
        gt_short = {g[:12] for g in gt_set}
        total_t += len(gt_set)
        cid_set  = set()                    # dedup across top-k lines

        for mg in ranked[:k]:
            mg_commits_short = {sha[:12] for sha in mg.tp_to_commit.values()}

            # Check if any commit in this line matches ground truth
            found_match = False
            for cid in mg_commits_short:
                if cid not in cid_set and cid in gt_short:
                    found_match = True
                    cid_set.add(cid)

            if found_match:
                tp += 1                     # one TP per deletion line
                continue

            # No GT match — check if any new (unseen) commit exists → FP
            has_new = False
            for cid in mg_commits_short:
                if cid not in cid_set:
                    cid_set.add(cid)
                    has_new = True
            if has_new:
                fp += 1

    precision = tp / (tp + fp) 
    recall    = tp / total_t 
    f1        = (2 * precision * recall / (precision + recall))

    return {
        f'precision@{k}': precision,
        f'recall@{k}':    recall,
        f'f1@{k}':        f1,
        f'tp@{k}':        tp,
        f'fp@{k}':        fp,
        'total_inducing_commits': total_t,
    }


def evaluate_top1_metrics(
    test_cases_graphs: Dict[str, List],
    true_cid_map: Optional[Dict[str, set]] = None,
    data_path: Optional[str] = None,
) -> Dict[str, float]:
    """Backward-compatible wrapper: evaluate at k=1."""
    return evaluate_topk_metrics(
        test_cases_graphs, true_cid_map, data_path, k=1)


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    """Pretty-print a metrics dict (supports @1 and optionally @2)."""
    tag = f"{prefix} " if prefix else ""
    print(f"{tag}Precision@1: {metrics['precision@1']:.4f} | "
          f"Recall@1: {metrics['recall@1']:.4f} | "
          f"F1@1: {metrics['f1@1']:.4f}")
    if "precision@2" in metrics:
        print(f"{tag}Precision@2: {metrics['precision@2']:.4f} | "
              f"Recall@2: {metrics['recall@2']:.4f} | "
              f"F1@2: {metrics['f1@2']:.4f}")


def compute_summary_statistics(all_metrics: List[Dict]) -> Dict[str, float]:
    """Average metrics across multiple folds."""
    if not all_metrics:
        return {"precision@1": 0.0, "recall@1": 0.0, "f1@1": 0.0}

    n = len(all_metrics)
    result = {
        "precision@1": sum(m["precision@1"] for m in all_metrics) / n,
        "recall@1":    sum(m["recall@1"]    for m in all_metrics) / n,
        "f1@1":        sum(m["f1@1"]        for m in all_metrics) / n,
    }
    if "precision@2" in all_metrics[0]:
        result["precision@2"] = sum(m.get("precision@2", 0) for m in all_metrics) / n
        result["recall@2"]    = sum(m.get("recall@2",    0) for m in all_metrics) / n
        result["f1@2"]        = sum(m.get("f1@2",        0) for m in all_metrics) / n
    if "commit_precision@1" in all_metrics[0]:
        result["commit_precision@1"] = sum(m.get("commit_precision@1", 0) for m in all_metrics) / n
        result["commit_recall@1"]    = sum(m.get("commit_recall@1",    0) for m in all_metrics) / n
        result["commit_f1@1"]        = sum(m.get("commit_f1@1",        0) for m in all_metrics) / n
    return result


def evaluate_ranking(
    model, dataset, test_cases: List[str], data_path: str,
    device: torch.device = None,
) -> Dict:
    """
    Score every deletion line in *dataset* and compute P/R/F1@1.

    This is the Phase 1 end-of-epoch evaluation routine.  It does NOT
    mutate the dataset's internal graph lists.
    """
    from training.utils import coerce_idx

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    graphs_dict = dataset.get_mini_graphs_dict()
    true_cid    = load_true_commit_map(test_cases, data_path)
    ranked: Dict = {}

    with torch.no_grad():
        for name in test_cases:
            if name not in graphs_dict or not graphs_dict[name]:
                continue
            for mg in graphs_dict[name]:
                try:
                    gd  = mg.pyg.to(device)
                    idx = coerce_idx(mg.del_idx)
                    mg.score = (
                        model.predict(gd, idx).item()
                        if idx < gd.num_nodes else 0.0
                    )
                except Exception:
                    mg.score = 0.0
            ranked[name] = sorted(
                graphs_dict[name], key=lambda g: g.score, reverse=True
            )

    return evaluate_top1_metrics(ranked, true_cid, data_path)



# Phase 2

def _prefix_to_full_sha_map(test_name: str, data_root: Path) -> dict:
    """Map 12-char SHA prefix → full SHA from commits.json / info.json."""
    root = data_root / test_name
    out: dict = {}
    for path in (root / "commits.json", root / "info.json"):
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for key in ("all_commits_in_history", "ground_truth", "induce",
                     "fix_commit", "fix", "vszz_introducer_commits"):
            if key not in data:
                continue
            vals = data[key]
            if isinstance(vals, str):
                vals = [vals]
            if not isinstance(vals, list):
                continue
            for sha in vals:
                if isinstance(sha, str) and len(sha) >= 12:
                    pfx = sha[:12]
                    if pfx not in out or len(sha) > len(out[pfx]):
                        out[pfx] = sha
    return out


def get_ranked_commit_shas(item, scores, top_k, expand_map):
    """Return deduplicated top-k predicted commit full SHAs from Phase 2 scores."""
    ranked_indices = torch.argsort(scores, descending=True).tolist()
    cmap = item.get("commit_idx_to_sha") or {}
    if not isinstance(cmap, dict):
        cmap = {}

    seen = set()
    shas = []
    for idx in ranked_indices[:top_k]:
        sha12 = cmap.get(int(idx), "")
        if sha12 and sha12 not in seen:
            full_sha = expand_map.get(sha12, sha12)
            shas.append(full_sha)
            seen.add(sha12)
    return shas


def evaluate_global(cases, item_map, true_cid_map, p2_model, device, data_root, k_values=(1, 2, 3)):
    """
    Global P/R/F1 at each @k across all cases in ``cases``.

    Every name in ``cases`` is counted once per ``k``: ``total_gt`` always
    includes ``len(true_cid_map[name])`` (possibly 0). If there is no valid
    Phase-2 item for a name, predictions are treated as empty (no model call):
    0 hits and 0 identified commits for that case — same effect as ranking
    that produced no pool to rank.
    """
    results_by_k = {}

    for k in k_values:
        total_hits = 0
        total_identified = 0
        total_gt = 0
        n_evaluated = 0

        with torch.no_grad():
            for name in cases:
                gt_set = true_cid_map.get(name, set())
                total_gt += len(gt_set)
                n_evaluated += 1

                item = item_map.get(name)
                if not item or not item.get("valid"):
                    # No pooled graph / invalid chain — same as no predicted commits
                    continue

                emb     = item['node_embeddings'].to(device)
                cidx    = item['commit_indices'].to(device)
                is_temp = item['is_temporal_node'].to(device)
                scores  = p2_model(emb, cidx, is_temp)
                if scores.dim() > 1:
                    scores = scores.squeeze(-1)
                if scores.dim() == 0:
                    scores = scores.unsqueeze(0)

                expand_map = _prefix_to_full_sha_map(name, data_root)
                preds = get_ranked_commit_shas(item, scores, k, expand_map)

                preds_set = set(preds)
                hits = len(preds_set & gt_set)

                total_hits += hits
                total_identified += len(preds_set)

        precision = total_hits / total_identified if total_identified > 0 else 0.0
        recall = total_hits / total_gt if total_gt > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        results_by_k[k] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_hits': total_hits,
            'total_identified': total_identified,
            'total_gt': total_gt,
            'n_evaluated': n_evaluated,
        }

   

    return results_by_k

