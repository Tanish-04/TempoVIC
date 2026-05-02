"""
Evaluation functions for the NeuralSZZ baseline.

Provides scoring, ranking, and metric computation for the HAN + RankNet model.
"""
import os
import json
import torch
from typing import Dict, List, Set
from functools import cmp_to_key

from config import GRAPH_DATA_DIR


def get_true_cid_map(fdirs):
    """Build mapping: composite_key -> set of true inducing commit IDs."""
    true_cid_map = {}
    for fdir in fdirs:
        true_cid_map[fdir] = set()
        info_path = GRAPH_DATA_DIR / fdir / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
                for cid in set(info.get("induce", [])):
                    true_cid_map[fdir].add(cid)
    return true_cid_map


def get_score(pyg, hanModel, rankNetModel, device):
    """Get the ranking score for a single mini graph."""
    with torch.no_grad():
        hanModel.eval()
        rankNetModel.eval()
        pyg = pyg.to(device)
        h = hanModel.predict(pyg, torch.tensor([0], device=device))[0]
        return rankNetModel.predict(h)


def cmp(x, y):
    """Compare two miniGraph objects by score (descending)."""
    score1 = x.score.double()
    score2 = y.score.double()
    if score1 < score2:
        return 1
    elif score1 > score2:
        return -1
    return 0


def score_and_rank(fdirs, dir_to_minigraphs, hanModel, rankNetModel, device):
    """Score all mini graphs and sort by descending score per test case."""
    hanModel.eval()
    rankNetModel.eval()
    with torch.no_grad():
        for fdir in fdirs:
            if fdir not in dir_to_minigraphs:
                continue
            for mini_graph in dir_to_minigraphs[fdir]:
                pyg = mini_graph.pyg
                score = get_score(pyg, hanModel, rankNetModel, device).to("cpu")
                mini_graph.score = score
            dir_to_minigraphs[fdir].sort(key=cmp_to_key(cmp))


def eval_top(fdirs, dir_to_minigraphs, hanModel, rankNetModel, device, true_cid_map, k):
    """Evaluate top-k precision: returns (tp, fp, total_true_commits)."""
    hanModel.eval()
    rankNetModel.eval()
    with torch.no_grad():
        tp = 0
        fp = 0
        total_t = 0
        for fdir in fdirs:
            if fdir not in dir_to_minigraphs:
                continue
            cidSet = set()
            total_t = total_t + len(true_cid_map.get(fdir, set()))
            for mini_graph in dir_to_minigraphs[fdir][:k]:
                node = mini_graph.g[0]

                f = False
                for cid in node["commits"]:
                    if cid not in cidSet and cid in true_cid_map.get(fdir, set()):
                        f = True
                        cidSet.add(cid)
                # the node is rootcause
                if f:
                    tp = tp + 1
                    continue

                # each node should correspond to one commit
                f1 = False
                for cid in node["commits"]:
                    if cid not in cidSet:
                        cidSet.add(cid)
                        f1 = True
                if f1:
                    fp = fp + 1

        return tp, fp, total_t


def eval_recall_topk(fdirs, dir_to_minigraphs, k):
    """Fraction of test cases where a rootcause node appears in top-k."""
    root_cause_cnt = 0
    valid_fdirs = [f for f in set(fdirs) if f in dir_to_minigraphs]
    if not valid_fdirs:
        return 0.0
    for fdir in valid_fdirs:
        for mini_graph in dir_to_minigraphs[fdir][:k]:
            node = mini_graph.g[0]
            if node["rootcause"]:
                root_cause_cnt += 1
                break
    return root_cause_cnt / len(valid_fdirs)


def eval_mean_first_rank(fdirs, dir_to_minigraphs):
    """Mean rank of the first rootcause node across test cases."""
    total_rank_cnt = 0
    valid_fdirs = [f for f in set(fdirs) if f in dir_to_minigraphs]
    if not valid_fdirs:
        return 0.0
    for fdir in valid_fdirs:
        for i, mini_graph in enumerate(dir_to_minigraphs[fdir]):
            node = mini_graph.g[0]
            if node["rootcause"]:
                total_rank_cnt = total_rank_cnt + i + 1
                break
    return total_rank_cnt / len(valid_fdirs)


def eval_top_metrics(
    fdirs: List[str],
    bic_commits_by_case: Dict[str, List[str]],
    true_cid_map: Dict[str, Set[str]],
) -> dict:
    """
    Pipeline-level Precision / Recall / F1 (micro-averaged over cases).

    Per case:
      preds_set = set of predicted commits (deduped within case)
      gt_set    = set of GT inducing commits
      hits      = |preds_set ∩ gt_set|

    Global:
      Precision = Σ hits / Σ |preds_set|
      Recall    = Σ hits / Σ |gt_set|
      F1        = harmonic mean
    """
    total_hits = 0
    total_identified = 0
    total_gt = 0
    n_no_bic = 0

    for fdir in fdirs:
        gt_set = true_cid_map.get(fdir, set())
        preds_set = set(bic_commits_by_case.get(fdir, []))

        if not preds_set:
            n_no_bic += 1

        hits = len(preds_set & gt_set)
        total_hits += hits
        total_identified += len(preds_set)
        total_gt += len(gt_set)

    precision = total_hits / total_identified if total_identified > 0 else 0.0
    recall = total_hits / total_gt if total_gt > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_hits": total_hits,
        "total_identified": total_identified,
        "total_gt": total_gt,
        "n_cases": len(fdirs),
        "n_no_bic": n_no_bic,
    }
