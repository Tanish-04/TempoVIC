"""
NeuralSZZ baseline training: HAN + RankNet with 70/15/15 split.

Usage:
    python train.py --seed 456
    python train.py --seed 456 --device cuda:1
"""
import argparse
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random

import numpy as np
import torch
import torch.nn as nn
import json
import sys
import gc
import copy

from itertools import chain
from sklearn.model_selection import train_test_split

from config import NEURALSZZ_DATA_DIR
from genPyG import *
from genPairs import *
from genBatch import *
from model import *
from eval import *
from genMiniGraphs import genAllMiniGraphs


def configure_seed(new_seed: int) -> None:
    global seed
    seed = int(new_seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

seed = 456
configure_seed(seed)


def get_all_data():
    """Load mini graphs and test case list from neuralszz/data/."""
    mini_graphs_path = NEURALSZZ_DATA_DIR / "miniGraphs_bszz.json"
    test_cases_path = NEURALSZZ_DATA_DIR / "test_cases.json"

    with open(mini_graphs_path) as f:
        all_miniGraphs = json.load(f)

    with open(test_cases_path) as f:
        all_test_cases = json.load(f)

    miniGraphs = {k: v for k, v in all_miniGraphs.items() if k in set(all_test_cases)}

    print(f"Loaded {len(all_miniGraphs)} total mini graphs, "
          f"filtered to {len(miniGraphs)} matching test cases")
    print(f"Total test cases: {len(all_test_cases)}")

    return miniGraphs, all_test_cases


def init_model(device, metadata):
    criterion = torch.nn.BCELoss()
    hanModel = HAN(
        device, 768, 768 * 2, metadata=metadata, heads=2,
        dropout=0.3, num_bert_layers_freeze=8,
    )
    hanModel = hanModel.to(device)
    rankNetModel = rankNet(768 * 2)
    rankNetModel = rankNetModel.to(device)

    bert_params = list(hanModel.bert_model.parameters())
    bert_ids = {id(p) for p in bert_params}
    rest_params = [p for p in chain(hanModel.parameters(), rankNetModel.parameters())
                   if id(p) not in bert_ids]

    optimizer = torch.optim.Adam([
        {"params": [p for p in bert_params if p.requires_grad], "lr": 2e-5},
        {"params": rest_params, "lr": 1e-4},
    ])

    return hanModel, rankNetModel, optimizer, criterion


def get_sub_minigraphs(fdirs, all_minigraphs):
    sub_minigraphs = {}
    for fdir in fdirs:
        if fdir in all_minigraphs:
            sub_minigraphs[fdir] = all_minigraphs[fdir]
    return sub_minigraphs


def get_all_batchlist(mini_graphs, k, max_pair=1000):
    all_batch_list = []
    pair_cnt = 0
    all_sub_minigraphs, all_sub_fdirs = divide_minigraphs(mini_graphs, k)

    for sub_minigraph in all_sub_minigraphs:
        all_pairs = get_all_pairs(sub_minigraph, max_pair)
        pair_cnt = pair_cnt + len(all_pairs)
        batch_list = combinePair(all_pairs, 128)
        all_batch_list.append(batch_list)

    return all_batch_list, all_sub_fdirs, pair_cnt


def divide_lst(lst, n, k):
    cnt = 0
    all_list = []
    for i in range(0, len(lst), n):
        if cnt < k - 1:
            all_list.append(lst[i : i + n])
        else:
            all_list.append(lst[i:])
            break
        cnt = cnt + 1
    return all_list


def divide_minigraphs(all_minigraphs, k):
    all_fdirs = list(all_minigraphs.keys())
    random.shuffle(all_fdirs)

    all_sub_minigraphs = []
    all_sub_fdirs = []
    for sub_fdirs in divide_lst(all_fdirs, int(len(all_fdirs) / k), k):
        if len(sub_fdirs) == 0:
            continue
        all_sub_fdirs.append(sub_fdirs)
        all_sub_minigraphs.append(get_sub_minigraphs(sub_fdirs, all_minigraphs))

    return all_sub_minigraphs, all_sub_fdirs


def train_batchlist(batches, hanModel, rankNetModel, optimizer, criterion, device):
    all_loss = []
    hanModel.train()
    rankNetModel.train()

    for batch_idx, batch in enumerate(batches):
        pyg1 = batch.pyg1.clone().to(device)
        pyg2 = batch.pyg2.clone().to(device)

        del_index1 = batch.del_index1.to(device)
        del_index2 = batch.del_index2.to(device)

        probs = batch.probs.to(device)
        x = hanModel(pyg1, del_index1)
        y = hanModel(pyg2, del_index2)

        optimizer.zero_grad()
        preds = rankNetModel(x, y)
        loss = criterion(preds, probs)
        loss.backward()
        optimizer.step()

        all_loss.append(loss.cpu().detach().item())

        # Explicit cleanup after each batch to prevent OOM
        del pyg1, pyg2, del_index1, del_index2, probs, x, y, preds, loss

        # Periodic CUDA cache clearing (every 50 batches)
        if batch_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return sum(all_loss)


def validate_batchlist(batches, hanModel, rankNetModel, criterion, device):
    all_loss = []
    hanModel.eval()
    rankNetModel.eval()

    for batch_idx, batch in enumerate(batches):
        with torch.no_grad():
            pyg1 = batch.pyg1.clone().to(device)
            pyg2 = batch.pyg2.clone().to(device)

            del_index1 = batch.del_index1.to(device)
            del_index2 = batch.del_index2.to(device)

            probs = batch.probs.to(device)
            x = hanModel(pyg1, del_index1)
            y = hanModel(pyg2, del_index2)

            preds = rankNetModel(x, y)
            loss = criterion(preds, probs)
            all_loss.append(loss.cpu().detach().item())

            del pyg1, pyg2, del_index1, del_index2, probs, x, y, preds, loss

        if batch_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return sum(all_loss)


def safe_f1(tp, fp, total_t):
    """Compute F1 score safely avoiding division by zero."""
    if tp + fp == 0 or total_t == 0 or tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / total_t
    return (2 * precision * recall) / (precision + recall)


def evaluate_split(split_name, split_cases, dir_to_minigraphs,
                   hanModel, rankNetModel, device, all_true_cid_map):
    """Evaluate on a split and return metrics dict."""
    score_and_rank(split_cases, dir_to_minigraphs, hanModel, rankNetModel, device)
    tp1, fp1, t = eval_top(split_cases, dir_to_minigraphs, hanModel, rankNetModel, device, all_true_cid_map, 1)
    tp2, fp2, t = eval_top(split_cases, dir_to_minigraphs, hanModel, rankNetModel, device, all_true_cid_map, 2)
    tp3, fp3, t = eval_top(split_cases, dir_to_minigraphs, hanModel, rankNetModel, device, all_true_cid_map, 3)
    total_t = t

    f1 = safe_f1(tp1, fp1, total_t)

    metrics = {
        f"{split_name}_f1_score": f1,
        f"{split_name}_top1_precision": tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0.0,
        f"{split_name}_top1_recall": tp1 / total_t if total_t > 0 else 0.0,
        f"{split_name}_top2_precision": tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0.0,
        f"{split_name}_top2_recall": tp2 / total_t if total_t > 0 else 0.0,
        f"{split_name}_top3_precision": tp3 / (tp3 + fp3) if (tp3 + fp3) > 0 else 0.0,
        f"{split_name}_top3_recall": tp3 / total_t if total_t > 0 else 0.0,
        f"{split_name}_recall@top1": eval_recall_topk(split_cases, dir_to_minigraphs, 1),
        f"{split_name}_recall@top2": eval_recall_topk(split_cases, dir_to_minigraphs, 2),
        f"{split_name}_recall@top3": eval_recall_topk(split_cases, dir_to_minigraphs, 3),
        f"{split_name}_mean_first_rank": eval_mean_first_rank(split_cases, dir_to_minigraphs),
    }
    return metrics


def do_train_val_test(device, split_seed: int = 456):
    """
    Train with fixed 70/15/15 split (sklearn, random_state=split_seed).
    """
    configure_seed(split_seed)

    all_mini_graphs, all_test_cases = get_all_data()

    # 70/15/15 split (random_state = split_seed)
    train_cases, temp = train_test_split(
        all_test_cases, test_size=0.3, random_state=split_seed)
    val_cases, test_cases = train_test_split(
        temp, test_size=0.5, random_state=split_seed)

    print(f"Train: {len(train_cases)}, Val: {len(val_cases)}, "
          f"Test: {len(test_cases)}")
    print(f"Random seed (split + RNG): {split_seed}")

    save_dir = f"./checkpoints_baseline_seed{split_seed}"
    results_dir = f"./results_baseline_seed{split_seed}"

    # Build training batches
    max_pair = 100
    mini_graphs = get_sub_minigraphs(train_cases, all_mini_graphs)
    all_batch_list, all_sub_fdirs, pair_cnt = get_all_batchlist(
        mini_graphs, 1, max_pair=max_pair
    )

    all_true_cid_map = get_true_cid_map(all_test_cases)
    dir_to_minigraphs = get_dir_to_minigraphs(
        get_sub_minigraphs(all_test_cases, all_mini_graphs)
    )

    # Init model
    hanModel, rankNetModel, optimizer, criterion = init_model(
        device, all_batch_list[0][0].pyg1.metadata()
    )

    # Training loop
    epochs = 20
    patience = 10
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    best_han_state = None
    best_rank_state = None

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    all_info = []

    print(f"\n{'='*70}")
    print(f"BASELINE TRAINING (train/val/test split, seed={split_seed})")
    print(f"{'='*70}")
    print(f"  Checkpoints → {save_dir}")
    print(f"  Results     → {results_dir}")
    print(f"Pairs: {pair_cnt}")

    for epoch in range(epochs):
        total_train_loss = train_batchlist(
            all_batch_list[0], hanModel, rankNetModel, optimizer, criterion, device
        )

        train_m = evaluate_split(
            "train", train_cases, dir_to_minigraphs,
            hanModel, rankNetModel, device, all_true_cid_map)

        val_m = evaluate_split(
            "val", val_cases, dir_to_minigraphs,
            hanModel, rankNetModel, device, all_true_cid_map)

        info = {"epoch": epoch, "pair_cnt": pair_cnt,
                "train_loss": total_train_loss}
        info.update(train_m)
        info.update(val_m)
        all_info.append(info)

        val_f1 = val_m["val_f1_score"]

        if epoch % 5 == 0 or epoch == 0:
            print(
                f"\n  Epoch {epoch}: train_loss={total_train_loss:.4f} | "
                f"train P@1={train_m['train_top1_precision']:.4f}, "
                f"F1={train_m['train_f1_score']:.4f} | "
                f"val P@1={val_m['val_top1_precision']:.4f}, "
                f"R@1={val_m['val_top1_recall']:.4f}, "
                f"F1={val_f1:.4f}")

        if val_f1 > best_val_f1 + 0.001:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_han_state = copy.deepcopy(hanModel.state_dict())
            best_rank_state = copy.deepcopy(rankNetModel.state_dict())

            torch.save(best_han_state, f"{save_dir}/han_best.pt")
            torch.save(best_rank_state, f"{save_dir}/ranknet_best.pt")

            print(f"  ✓ New best (F1={val_f1:.4f}, epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⚠ Early stopping at epoch {epoch}")
                break

        # Save epoch metrics
        with open(f"{results_dir}/training_log.json", "w") as f:
            json.dump(all_info, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    # Load best model & evaluate on test
    print(f"\n  Best epoch: {best_epoch}, Val F1@1: {best_val_f1:.4f}")
    print(f"  Evaluating on held-out test set ({len(test_cases)} cases)...")

    if best_han_state is not None:
        hanModel.load_state_dict(best_han_state)
        rankNetModel.load_state_dict(best_rank_state)

    test_m = evaluate_split(
        "test", test_cases, dir_to_minigraphs,
        hanModel, rankNetModel, device, all_true_cid_map)

    print(f"\n{'='*70}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*70}")
    print(f"  P@1  = {test_m['test_top1_precision']:.4f}")
    print(f"  R@1  = {test_m['test_top1_recall']:.4f}")
    print(f"  F1@1 = {test_m['test_f1_score']:.4f}")
    print(f"  P@2  = {test_m['test_top2_precision']:.4f}")
    print(f"  P@3  = {test_m['test_top3_precision']:.4f}")
    print(f"  MFR  = {test_m['test_mean_first_rank']:.4f}")
    print(f"  Recall@1 = {test_m['test_recall@top1']:.4f}")
    print(f"  Recall@2 = {test_m['test_recall@top2']:.4f}")
    print(f"  Recall@3 = {test_m['test_recall@top3']:.4f}")

    # Save final results
    final_results = {
        "seed": split_seed,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_metrics": test_m,
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "test_cases": len(test_cases),
        "save_dir": save_dir,
        "results_dir": results_dir,
    }
    with open(f"{results_dir}/final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n✓ Results saved to {results_dir}/final_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuralSZZ baseline (HAN + RankNet) — 70/15/15 train/val/test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=456,
        help="Random seed for split + RNG (default: 456)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda:0, cuda:1, or cpu (default: cuda:0 if CUDA available)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed:   {args.seed}")

    print("Starting baseline training with train/val/test split...")
    do_train_val_test(device, split_seed=args.seed)