"""
training/phase2_trainer.py

Phase 2 training loop: commit ranking on pre-computed node embeddings.

The frozen Phase 1 encoder has already been applied during the embedding
pre-computation step (training/embedding_cache.py).  This trainer only
optimises CommitRankingModule.

Evaluation uses evaluate_global() which computes global P/R/F1 at
k=1,2,3 across all cases — supporting top-3 VIC ranking.
"""

import copy
import gc
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset

from data_processing.dataset import collate_commit_ranking
from models.phase2_model import CommitRankingModule
from training.loss import LabelSmoothingRankingLoss
from training.utils import (
    EarlyStopping,
    build_phase2_model,
    clip_and_step,
    setup_device,
)
from training.evaluation import evaluate_global, load_true_commit_map


# ── Epoch runner (loss only — metrics via evaluate_global) ────────────────────

def _run_epoch(
    model: CommitRankingModule,
    loader: DataLoader,
    optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    *,
    training: bool,
    log_interval: int = 50,
    grad_accum: int = 1,
) -> float:
    """
    Run one training or validation epoch (loss computation only).

    Metrics are computed separately via ``evaluate_global`` after the
    epoch finishes, giving global P/R/F1 at k=1,2,3.

    Args:
        training     : True  -> model.train(), backprop, gradient accumulation.
                       False -> model.eval(), torch.no_grad(), no backprop.
        log_interval : print a running loss summary every N batches (training only).
        grad_accum   : gradient accumulation steps (training only).

    Returns:
        avg_loss (float)
    """
    if training:
        model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        model.eval()

    total_loss, total_samples = 0.0, 0
    t0 = time.time()

    grad_ctx = nullcontext() if training else torch.no_grad()
    with grad_ctx:
        for b_idx, batch in enumerate(loader):
            if not batch:
                continue

            try:
                scores_all = model(
                    batch["node_embeddings"].to(device),
                    batch["commit_indices"].to(device),
                    batch["is_temporal_node"].to(device),
                )
                scores_list = torch.split(scores_all, batch["commit_counts"])

                b_loss  = torch.tensor(0.0, device=device)
                b_count = 0

                for scores, gt in zip(scores_list, batch["ground_truth_positions"]):
                    if not gt:
                        continue

                    loss = loss_fn(scores, gt)
                    if training:
                        loss = loss / grad_accum
                    if not torch.isnan(loss):
                        b_loss  = b_loss + loss
                        b_count += 1

                if b_count:
                    total_loss    += b_loss.item() * (grad_accum if training else 1)
                    total_samples += b_count
                    if training:
                        b_loss.backward()

            except Exception as exc:
                phase = "Train" if training else "Val"
                print(f"  [{phase}] batch {b_idx}: {type(exc).__name__}: {exc}")
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()

            if training:
                if (b_idx + 1) % grad_accum == 0:
                    clip_and_step(model, optimizer)

                if (b_idx + 1) % log_interval == 0:
                    print(
                        f"  Batch {b_idx+1}/{len(loader)}: "
                        f"loss={total_loss/max(total_samples,1):.4f}, "
                        f"t={time.time()-t0:.1f}s"
                    )

    # Flush any remaining accumulated gradients at end of training epoch
    if training and total_samples and len(loader) % grad_accum != 0:
        clip_and_step(model, optimizer)

    return total_loss / max(total_samples, 1)


# Public fold trainer 

def train_phase2_fold(
    fold_idx: int,
    train_indices: List[int],
    val_indices: List[int],
    phase2_dataset,
    config: Dict,
    *,
    train_cases: List[str],
    val_cases: List[str],
    item_map: Dict[str, dict],
    true_cid_map: Dict[str, set],
    data_root: Path,
) -> Dict:
    """
    Train Phase 2 (commit ranking) on pre-computed embeddings.

    Parameters
    ----------
    train_cases / val_cases : test-case names for evaluate_global
    item_map     : {test_name: item_dict} built from the Phase 2 dataset
    true_cid_map : {test_name: set_of_inducing_shas} from load_true_commit_map
    data_root    : Path to trainData/ (needed by evaluate_global for SHA expansion)

    Returns
    -------
    dict with keys:
        fold                     : int
        best_epoch               : int
        best_val_f1@1            : float
        final_metrics            : Dict
        history                  : Dict[str, List[float]]
        best_commit_ranker_state : state_dict of CommitRankingModule
    """
    print(f"\n{'─'*60}")
    print(f"  PHASE 2 — Fold {fold_idx + 1}: Commit Ranking")
    print(f"{'─'*60}")

    device = setup_device(config["defaults"].get("gpu_id", 0))
    g = torch.Generator()
    g.manual_seed(config["defaults"]["seed"])
    
    loader_kw = dict(
        collate_fn=collate_commit_ranking,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        generator=g
    )

    train_loader = DataLoader(
        Subset(phase2_dataset, train_indices),
        batch_size=config["phase2"]["batch_size"],
        shuffle=True,
        **loader_kw,
    )
    val_loader = DataLoader(
        Subset(phase2_dataset, val_indices),
        batch_size=config["phase2"]["batch_size"],
        shuffle=False,
        **loader_kw,
    )

    model = build_phase2_model(config, device)
    print(f"  Device: {device}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(
        model.parameters(),
        lr=config["phase2"]["lr"],
        weight_decay=config["phase2"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)
    loss_fn   = LabelSmoothingRankingLoss(
        temperature=config["phase2"]["temperature"],
        margin=config["phase2"]["margin"],
        smoothing=config["phase2"]["label_smoothing"],
        focal_gamma=config["phase2"]["focal_gamma"],
        focal_alpha=config["phase2"]["focal_alpha"],
    )
   

    stopper = EarlyStopping(patience=config["phase2"]["patience"], mode="max")
    ga      = config["phase2"]["gradient_accumulation_steps"]

    best_f1, best_epoch         = 0.0, 0
    best_state: Optional[Dict]  = None
    best_metrics: Optional[Dict] = None
    history = {
        "train_loss": [], "val_loss": [],
        "train_f1@1": [], "val_f1@1": [],
    }

    for epoch in range(1, config["phase2"]["epochs"] + 1):
        t0 = time.time()

        tr_loss = _run_epoch(
            model, train_loader, optimizer, loss_fn, device,
            training=True, log_interval=config["defaults"]["log_interval"], grad_accum=ga)

        vl_loss = _run_epoch(
            model, val_loader, optimizer, loss_fn, device,
            training=False
        )

        # Evaluate with evaluate_global (global P/R/F1 at k=1,2,3)
        val_results = evaluate_global(
            cases=val_cases,
            item_map=item_map,
            true_cid_map=true_cid_map,
            p2_model=model,
            device=device,
            data_root=data_root,
        )
        # Use k=1 F1 for model selection and early stopping
        vl_f1 = val_results.get(1, {}).get("f1", 0.0)
        scheduler.step(vl_f1)
        # scheduler.step(epoch)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_f1@1"].append(vl_f1)

        if epoch % 5 == 0 or epoch == 1:
            vl_k1 = val_results.get(1, {})
            print(
                f"\n  Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={vl_loss:.4f} | "
                f"val P@1={vl_k1.get('precision',0):.4f}, "
                f"R@1={vl_k1.get('recall',0):.4f}, "
                f"F1@1={vl_k1.get('f1',0):.4f}  [{time.time()-t0:.1f}s]"
            )

        if vl_f1 > best_f1:
            best_f1, best_epoch = vl_f1, epoch
            best_state   = copy.deepcopy(model.state_dict())
            best_metrics = val_results
            print(f"  ✓ New best (F1@1={vl_f1:.4f}, epoch {epoch})")

        if stopper(vl_f1, epoch):
            print(f"  ⚠ Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    final_m = best_metrics or {}

    # Print best results at all k values
    print(f"\n  Fold {fold_idx+1} best (epoch {best_epoch}):")
    for k in sorted(final_m.keys()):
        km = final_m[k]
        print(
            f"    @{k}: P={km.get('precision',0):.4f}, "
            f"R={km.get('recall',0):.4f}, "
            f"F1={km.get('f1',0):.4f}"
        )

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "fold":                     fold_idx + 1,
        "best_epoch":               best_epoch,
        "best_val_f1@1":            best_f1,
        "final_metrics":            final_m,
        "history":                  history,
        "best_commit_ranker_state": best_state,
    }