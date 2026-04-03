import os
import sys
import json
import time
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
from contextlib import contextmanager  # Added for nullcontext

try:
    # PyTorch builds exposing unified AMP API.
    from torch.amp import autocast, GradScaler
    AMP_SUPPORTS_DEVICE_TYPE = True
except ImportError:
    # Fallback for older PyTorch builds.
    from torch.cuda.amp import autocast, GradScaler
    AMP_SUPPORTS_DEVICE_TYPE = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.transformer.dataset import SASRecTrainDataset, SASRecEvalDataset, load_datasets
from models.transformer.model   import SASRecModel, sampled_bce_loss

# ─────────────────────────────────────────────────────────────
# CONTEXT MANAGER SHIM (moved to top to avoid NameError)
# ─────────────────────────────────────────────────────────────

@contextmanager
def nullcontext():
    yield

# ─────────────────────────────────────────────────────────────
# HYPERPARAMETERS — RTX 3050 tuned
# ─────────────────────────────────────────────────────────────

# Model config
N_ITEMS     = None      # loaded from dataset_meta.json
HIDDEN_DIM  = 64        # 128 if you have spare VRAM (test with model smoke test first)
NUM_HEADS   = 2         # hidden_dim / num_heads = 32 per head
NUM_BLOCKS  = 2         # Transformer depth
MAX_SEQ_LEN = 50        # from Week 1
DROPOUT     = 0.2

# Training config
BATCH_SIZE  = 256       # safe for 4GB VRAM with fp16
NEG_SAMPLES = 128       # negatives per position
EPOCHS      = 20
LR          = 1e-3      # Adam learning rate
WEIGHT_DECAY= 1e-4      # L2 regularization
MAX_GRAD_NORM= 5.0      # gradient clipping (higher than MF — Transformers need more)
USE_AMP     = True      # Automatic Mixed Precision (set False only if issues arise)

# Evaluation config
EVAL_EVERY  = 1         # evaluate every N epochs
EVAL_K      = [5, 10, 20]
TOP_K_METRIC= 10        # primary metric: HR@10 for model selection

# Checkpointing
PATIENCE    = 5         # early stopping patience (HR@10)
SAVE_EVERY  = 5         # save checkpoint every N epochs

# Paths
MODELS_DIR  = Path("models/saved")
LOGS_DIR    = Path("logs")
PROCESSED_DIR = Path("data/processed")

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)

def save_checkpoint(model, optimizer, scaler, epoch, metrics, path):
    torch.save({
        "epoch"           : epoch,
        "model_state"     : model.state_dict(),
        "optimizer_state" : optimizer.state_dict(),
        "scaler_state"    : scaler.state_dict() if scaler else None,
        "metrics"         : metrics,
        "hyperparams"     : {
            "n_items"    : model.n_items,
            "hidden_dim" : model.hidden_dim,
            "max_seq_len": model.max_seq_len,
            "num_blocks" : model.num_blocks,
        },
    }, path)

def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler and ckpt.get("scaler_state"):
        scaler.load_state_dict(ckpt["scaler_state"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})

# ─────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────

def train_epoch(
    model:     SASRecModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler:    Optional[GradScaler],
    device:    torch.device,
    use_amp:   bool,
    epoch:     int,
) -> Tuple[float, float]:
    """
    Runs one full training epoch.
    Returns (avg_loss, epoch_time_seconds).

    RTX 3050 note:
      autocast() wraps only the forward + loss computation.
      scaler.scale(loss).backward() handles fp16 gradient scaling.
      scaler.unscale_(optimizer) + clip_grad_norm_ + scaler.step() is the
      correct order — do NOT change this sequence.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t_start    = time.time()

    for batch_idx, batch in enumerate(loader):
        seq = batch["seq"].to(device, non_blocking=True)   # [B, L]
        pos = batch["pos"].to(device, non_blocking=True)   # [B, L]
        neg = batch["neg"].to(device, non_blocking=True)   # [B, L, K]

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        # ── Forward pass with AMP ──────────────────────────────
        if use_amp:
            if AMP_SUPPORTS_DEVICE_TYPE:
                amp_ctx = autocast(device_type='cuda')
            else:
                amp_ctx = autocast()

            with amp_ctx:
                hidden = model(seq)                          # [B, L, D]
                loss   = sampled_bce_loss(hidden, pos, neg, model.item_emb)

            if not torch.isfinite(loss):
                continue

            # ── Backward pass with gradient scaling ───────────
            scaler.scale(loss).backward()
            # Unscale before clipping (required for correct norms)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            # fp32 fallback (CPU or if AMP disabled)
            hidden = model(seq)
            loss   = sampled_bce_loss(hidden, pos, neg, model.item_emb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        # ── Progress print every 200 batches ──────────────────
        if (batch_idx + 1) % 200 == 0:
            avg = total_loss / n_batches
            elapsed = time.time() - t_start
            batches_left = len(loader) - (batch_idx + 1)
            eta = elapsed / (batch_idx + 1) * batches_left
            print(f"    Epoch {epoch:2d} | "
                  f"Batch {batch_idx+1:5d}/{len(loader)} | "
                  f"Loss: {avg:.4f} | "
                  f"ETA: {eta:.0f}s")

    avg_loss  = total_loss / max(n_batches, 1)
    epoch_time= time.time() - t_start
    return avg_loss, epoch_time


# ─────────────────────────────────────────────────────────────
# QUICK EVAL (during training, uses sampled negatives)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def quick_eval_hr10(
    model:   SASRecModel,
    val_ds:  SASRecEvalDataset,
    device:  torch.device,
    k:       int = 10,
    n_users: int = 2000,   # evaluate on random 2000 users (faster than full val)
    n_neg:   int = 100,    # 100 sampled negatives + 1 positive
) -> float:
    """
    Fast HR@K during training using sampled negatives (not full item ranking).

    Methodology (standard in SASRec paper):
      - For each user: rank 1 positive against 100 random negatives
      - HR@K = fraction of users where positive is in top K

    NOT used for final evaluation (use full ranking for that).
    Fast enough to run every epoch for model selection.
    """
    model.eval()

    n_users = min(n_users, len(val_ds))
    indices = torch.randperm(len(val_ds))[:n_users].tolist()

    hits    = 0
    total   = 0

    loader = DataLoader(
        torch.utils.data.Subset(val_ds, indices),
        batch_size = 512,
        shuffle    = False,
        num_workers= 0,
    )

    meta    = json.load(open(PROCESSED_DIR / "dataset_meta.json"))
    n_items = meta.get("n_items_with_pad", meta["n_items"] + 1)

    for batch in loader:
        seq   = batch["seq"].to(device)    # [B, L]
        label = batch["label"].to(device)  # [B]
        B     = seq.shape[0]

        # Sample n_neg negatives per user (random, not checking history)
        negs  = torch.randint(1, n_items, (B, n_neg), device=device)  # [B, n_neg]

        # Build candidates: [label] + [negs] → [B, n_neg+1]
        candidates = torch.cat([label.unsqueeze(1), negs], dim=1)  # [B, n_neg+1]

        # Score candidates
        scores = model.predict_scores(seq, candidates)   # [B, n_neg+1]

        # Rank: position 0 is the positive
        ranks  = (scores >= scores[:, 0:1]).sum(dim=1)   # rank of positive (1-indexed)
        hits  += (ranks <= k).sum().item()
        total += B

    return hits / max(total, 1)


# ─────────────────────────────────────────────────────────────
# FULL TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train(
    n_items:     int,
    train_ds:    SASRecTrainDataset,
    val_ds:      SASRecEvalDataset,
    use_mlflow:  bool = True,
) -> SASRecModel:

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = USE_AMP and device.type == "cuda"

    print_section("DS19 — Week 3 | SASRec Training")
    print(f"\n  Device        : {device}")
    print(f"  AMP (fp16)    : {use_amp}")
    print(f"  n_items       : {n_items:,}")
    print(f"  HIDDEN_DIM    : {HIDDEN_DIM}")
    print(f"  NUM_HEADS     : {NUM_HEADS}")
    print(f"  NUM_BLOCKS    : {NUM_BLOCKS}")
    print(f"  BATCH_SIZE    : {BATCH_SIZE}")
    print(f"  NEG_SAMPLES   : {NEG_SAMPLES}")
    print(f"  EPOCHS        : {EPOCHS}")
    print(f"  LR            : {LR}")
    print(f"  WEIGHT_DECAY  : {WEIGHT_DECAY}")
    print(f"  MAX_GRAD_NORM : {MAX_GRAD_NORM}")
    print(f"  PATIENCE      : {PATIENCE}")

    # ── DataLoader ────────────────────────────────────────────
    # Windows note: if num_workers > 0 causes issues, set to 0
    num_workers = 2 if os.name != 'nt' else 0
    train_loader = DataLoader(
        train_ds,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,   # avoids batch-size-1 edge cases with LayerNorm
    )
    print(f"\n  Train batches : {len(train_loader):,}")
    print(f"  Val users     : {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────
    model = SASRecModel(
        n_items    = n_items,
        hidden_dim = HIDDEN_DIM,
        max_seq_len= MAX_SEQ_LEN,
        num_heads  = NUM_HEADS,
        num_blocks = NUM_BLOCKS,
        dropout    = DROPOUT,
    ).to(device)

    print(f"\n  Parameters    : {model.count_parameters():,}")

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = LR,
        weight_decay = WEIGHT_DECAY,
        betas        = (0.9, 0.999),
        eps          = 1e-8,
    )

    # ── LR Scheduler ──────────────────────────────────────────
    # Cosine annealing: smoothly decays LR over training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = EPOCHS,
        eta_min= LR / 20,   # final LR = initial / 20
    )

    # ── AMP Gradient Scaler ───────────────────────────────────
    if use_amp:
        scaler = GradScaler("cuda") if AMP_SUPPORTS_DEVICE_TYPE else GradScaler()
    else:
        scaler = None

    # ── MLflow ────────────────────────────────────────────────
    mlflow_available = False
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("DS19_SASRec")
            mlflow_available = True
            print(f"\n  MLflow        : enabled (http://localhost:5000)")
        except Exception as e:
            print(f"\n  MLflow        : disabled ({e})")

    # ── Training state ────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    best_hr10     = 0.0
    patience_ctr  = 0
    history       = []   # per-epoch metrics

    # ── MLflow run ────────────────────────────────────────────
    run_context = mlflow.start_run(run_name=f"sasrec_{datetime.now().strftime('%m%d_%H%M')}") \
                  if mlflow_available else nullcontext()

    with run_context:
        if mlflow_available:
            mlflow.log_params({
                "model"      : "SASRec",
                "n_items"    : n_items,
                "hidden_dim" : HIDDEN_DIM,
                "num_heads"  : NUM_HEADS,
                "num_blocks" : NUM_BLOCKS,
                "dropout"    : DROPOUT,
                "batch_size" : BATCH_SIZE,
                "neg_samples": NEG_SAMPLES,
                "lr"         : LR,
                "epochs"     : EPOCHS,
                "use_amp"    : use_amp,
            })

        # ── Epoch loop ────────────────────────────────────────
        for epoch in range(1, EPOCHS + 1):
            print(f"\n  {'─'*60}")
            print(f"  Epoch {epoch}/{EPOCHS}")
            print(f"  {'─'*60}")

            # ── Train ─────────────────────────────────────────
            avg_loss, epoch_time = train_epoch(
                model, train_loader, optimizer, scaler, device, use_amp, epoch
            )

            # ── Quick val HR@10 ───────────────────────────────
            hr10 = quick_eval_hr10(model, val_ds, device, k=TOP_K_METRIC)
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"\n  → Loss: {avg_loss:.4f} | "
                  f"HR@{TOP_K_METRIC}: {hr10:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.0f}s")

            # ── LR step ───────────────────────────────────────
            scheduler.step()

            # ── Log to MLflow ─────────────────────────────────
            if mlflow_available:
                mlflow.log_metrics({
                    "train_loss": avg_loss,
                    f"hr_{TOP_K_METRIC}_sampled": hr10,
                    "lr": current_lr,
                }, step=epoch)

            # ── Record history ────────────────────────────────
            history.append({
                "epoch"     : epoch,
                "train_loss": round(avg_loss, 6),
                f"hr{TOP_K_METRIC}_sampled": round(hr10, 6),
                "lr"        : round(current_lr, 8),
                "time_s"    : round(epoch_time, 1),
            })

            # ── Save checkpoint every SAVE_EVERY epochs ───────
            if epoch % SAVE_EVERY == 0:
                ckpt_path = MODELS_DIR / f"sasrec_epoch_{epoch:03d}.pt"
                save_checkpoint(model, optimizer, scaler, epoch, {"hr10": hr10}, ckpt_path)
                print(f"  💾 Checkpoint saved: {ckpt_path.name}")

            # ── Best model tracking ────────────────────────────
            if hr10 > best_hr10:
                best_hr10   = hr10
                patience_ctr = 0
                best_path    = MODELS_DIR / "sasrec_best.pt"
                save_checkpoint(model, optimizer, scaler, epoch, {"hr10": hr10, "epoch": epoch}, best_path)
                print(f"  🏆 New best HR@{TOP_K_METRIC}: {best_hr10:.4f} → saved sasrec_best.pt")
            else:
                patience_ctr += 1
                print(f"  ⏳ No improvement ({patience_ctr}/{PATIENCE})")

            # ── Early stopping ────────────────────────────────
            if patience_ctr >= PATIENCE:
                print(f"\n  🛑 Early stopping at epoch {epoch} "
                      f"(no HR@{TOP_K_METRIC} improvement for {PATIENCE} epochs)")
                break

            # ── Clear GPU cache ────────────────────────────────
            # Helps prevent VRAM fragmentation on RTX 3050
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── Save final epoch model ─────────────────────────────────
    final_path = MODELS_DIR / f"sasrec_epoch_{epoch:03d}_final.pt"
    save_checkpoint(model, optimizer, scaler, epoch, {"hr10": hr10}, final_path)
    print(f"\n  💾 Final model saved: {final_path.name}")

    # ── Save training history ──────────────────────────────────
    import csv
    log_path = LOGS_DIR / "sasrec_training_log.csv"
    if history:
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=history[0].keys())
            w.writeheader()
            w.writerows(history)
    print(f"  📊 Training log saved: {log_path}")

    print_section("Training Complete")
    print(f"  Best HR@{TOP_K_METRIC} (sampled, 100 neg): {best_hr10:.4f}")
    print(f"  Best model: models/saved/sasrec_best.pt")
    print(f"  Run full evaluation: python models/transformer/evaluate.py")

    return model


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_section("Loading datasets...")

    train_ds, val_ds, test_ds, meta = load_datasets(
        neg_samples = NEG_SAMPLES,
        max_seq_len = MAX_SEQ_LEN,
    )
    n_items = meta.get("n_items_with_pad", meta["n_items"] + 1)

    model = train(
        n_items    = n_items,
        train_ds   = train_ds,
        val_ds     = val_ds,
        use_mlflow = True,
    )