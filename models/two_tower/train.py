import sys
import json
import time
import pickle
import math
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import gc
import os

try:
    from torch.amp import autocast, GradScaler
    AMP_SUPPORTS_DEVICE_TYPE = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_SUPPORTS_DEVICE_TYPE = False

# ─────────────────────────────────────────────────────────────
# LOCAL IMPORTS
# ─────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from models.two_tower.dataset import TwoTowerTrainDataset, TwoTowerEvalDataset
from models.two_tower.model   import TwoTowerModel, infonce_loss

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR    = Path("data/splits")
MODELS_DIR    = Path("models/saved")
LOGS_DIR      = Path("logs")

# Model
EMBED_DIM     = 64
NUM_HEADS     = 2
NUM_BLOCKS    = 2
DROPOUT       = 0.2
MAX_SEQ_LEN   = 50
PAD_TOKEN     = 0
TEMPERATURE   = 0.07

# Training - OPTIMIZED for 4GB VRAM / 16GB RAM
BATCH_SIZE    = 128
EPOCHS        = 10
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0
SEED          = 42
USE_AMP       = True

# GPU throttle
GPU_MEMORY_FRACTION = 0.75

# Evaluation (during training — sampled)
EVAL_EVERY    = 2
EVAL_TOPK     = 50
NEG_EVAL      = 99

# Checkpointing
SAVE_EVERY    = 5

# MLflow
USE_MLFLOW    = False

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        vram_capped = vram_total * GPU_MEMORY_FRACTION
        print(f"  VRAM total   : {vram_total:.1f} GB")
        print(f"  VRAM cap (75%): {vram_capped:.1f} GB")

        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=0)
        torch.cuda.empty_cache()

        if vram_total < 5.0:
            print(f"  ⚠️  4GB VRAM detected — BATCH_SIZE set to {BATCH_SIZE}")
    else:
        device = torch.device("cpu")
        print("  ⚠️  Using CPU — training will be slow")
    return device

def load_json_safe(path):
    """Load JSON and immediately close file handle"""
    with open(path, "r") as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────
# SAMPLED RECALL@K - MEMORY OPTIMIZED
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def sampled_recall_at_k(model: TwoTowerModel,
                         val_ds,
                         device: torch.device,
                         k: int = EVAL_TOPK,
                         n_neg: int = NEG_EVAL,
                         n_users: int = 1000) -> float:
    """
    Memory-optimized sampled Recall@K.
    """
    model.eval()
    
    meta = load_json_safe(PROCESSED_DIR / "dataset_meta.json")
    n_items = meta['n_items']

    n_eval = min(n_users, len(val_ds))
    hits = 0
    
    micro_batch_size = 100
    
    for start_idx in range(0, n_eval, micro_batch_size):
        end_idx = min(start_idx + micro_batch_size, n_eval)
        
        for idx in range(start_idx, end_idx):
            seq, label = val_ds[idx]
            seq = seq.unsqueeze(0).to(device)
            label = label.item()

            user_emb = model.encode_user(seq)

            neg_candidates = torch.randint(1, n_items + 1, (n_neg * 2,))
            neg_candidates = neg_candidates[neg_candidates != label][:n_neg]
            if len(neg_candidates) < n_neg:
                additional = torch.randint(1, n_items + 1, (n_neg,))
                neg_candidates = torch.cat([neg_candidates, additional])[:n_neg]

            all_ids = torch.cat([torch.tensor([label]), neg_candidates]).long().to(device)
            item_embs = model.encode_item(all_ids)

            scores = (user_emb @ item_embs.T).squeeze(0)
            label_score = scores[0].item()
            rank = (scores > label_score).sum().item() + 1

            if rank <= k:
                hits += 1

            del seq, user_emb, all_ids, item_embs, scores
            
        torch.cuda.empty_cache()
        gc.collect()

    model.train()
    return hits / n_eval

# ─────────────────────────────────────────────────────────────
# TRAINING - OPTIMIZED FOR BETTER GPU UTILIZATION
# ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_ds, optimizer, scaler,
          scheduler, device, n_items, start_epoch, best_recall):
    """Training loop optimized for 4GB VRAM utilization."""

    log_path = LOGS_DIR / "two_tower_training_log.csv"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "sampled_recall_50", "lr", "time_min"])

    best_epoch = start_epoch

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        epoch_start = time.time()
        total_loss = 0.0
        total_batches = 0
        model.train()

        for batch_idx, (seqs, targets) in enumerate(train_loader):
            seqs = seqs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            if AMP_SUPPORTS_DEVICE_TYPE:
                amp_ctx = autocast(device_type='cuda', enabled=USE_AMP)
            else:
                amp_ctx = autocast(enabled=USE_AMP)

            with amp_ctx:
                user_embs, item_embs = model(seqs, targets)
                loss = infonce_loss(user_embs, item_embs, temperature=TEMPERATURE)

            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            del seqs, targets, user_embs, item_embs, loss
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()

            if batch_idx % 50 == 0:
                avg = total_loss / total_batches
                pct = 100.0 * batch_idx / len(train_loader)
                alloc = torch.cuda.memory_allocated(device) / 1e9
                reserv = torch.cuda.memory_reserved(device) / 1e9
                # ← FIXED: Removed torch.cuda.utilization() call
                print(
                    f"    Epoch {epoch:02d} [{pct:5.1f}%]  "
                    f"loss={avg:.4f}  "
                    f"GPU alloc={alloc:.2f}GB / reserved={reserv:.2f}GB",
                    end="\r"
                )

        torch.cuda.empty_cache()
        gc.collect()

        scheduler.step()
        avg_loss = total_loss / total_batches
        epoch_min = (time.time() - epoch_start) / 60
        lr_now = scheduler.get_last_lr()[0]

        recall_50 = 0.0
        if epoch % EVAL_EVERY == 0:
            print(f"\n    Epoch {epoch:02d} — Running sampled Recall@{EVAL_TOPK}...")
            recall_50 = sampled_recall_at_k(model, val_ds, device, k=EVAL_TOPK)
            is_best = recall_50 > best_recall

            if is_best:
                best_recall = recall_50
                best_epoch = epoch
                model_cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({
                    'epoch': epoch,
                    'model_state': model_cpu_state,
                    'optimizer': optimizer.state_dict(),
                    'best_recall': best_recall,
                    'config': {
                        'n_items': n_items,
                        'hidden_dim': model.hidden_dim,
                        'num_heads': NUM_HEADS,
                        'num_blocks': NUM_BLOCKS,
                        'max_seq_len': MAX_SEQ_LEN,
                        'dropout': DROPOUT,
                        'temperature': TEMPERATURE,
                    }
                }, MODELS_DIR / "two_tower_best.pt")
                del model_cpu_state
                torch.cuda.empty_cache()
                print(f"    ⭐ New best Recall@{EVAL_TOPK}: {best_recall:.4f} → saved two_tower_best.pt")

        if epoch % SAVE_EVERY == 0:
            ckpt_path = MODELS_DIR / f"two_tower_epoch_{epoch:03d}.pt"
            model_cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({'epoch': epoch, 'model_state': model_cpu_state}, ckpt_path)
            del model_cpu_state
            torch.cuda.empty_cache()

        print(f"  Epoch {epoch:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Recall@{EVAL_TOPK}: {recall_50:.4f} | "
              f"LR: {lr_now:.2e} | {epoch_min:.1f} min")

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(avg_loss, 6), round(recall_50, 6),
                             round(lr_now, 8), round(epoch_min, 2)])

    return best_recall, best_epoch

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    global USE_MLFLOW
    print_section("DS19 — Week 4 | Two-Tower Training (4GB VRAM Optimized)")
    set_seed(SEED)
    device = get_device()

    meta = load_json_safe(PROCESSED_DIR / "dataset_meta.json")
    n_items = meta['n_items']
    print(f"\n  n_items: {n_items:,}")

    print_section("Loading Datasets")
    train_ds = TwoTowerTrainDataset(max_seq_len=MAX_SEQ_LEN)
    val_ds = TwoTowerEvalDataset(split="val", max_seq_len=MAX_SEQ_LEN)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    print(f"\n  Training samples : {len(train_ds):,}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  In-batch negatives per positive: {BATCH_SIZE - 1}")
    print(f"  Batches per epoch: {len(train_loader):,}")
    print(f"  Expected VRAM    : ~1.5-2.0GB")
    print(f"  Num workers      : 0")
    print(f"  Pin memory       : False")

    print_section("Building Model")
    model = TwoTowerModel(
        n_items=n_items,
        hidden_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT,
    ).to(device)

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("  ✅  Gradient checkpointing enabled")

    print(f"  Parameters     : {model.count_parameters():,}")
    print(f"  EMBED_DIM      : {EMBED_DIM}")
    print(f"  NUM_BLOCKS     : {NUM_BLOCKS}")
    print(f"  TEMPERATURE    : {TEMPERATURE}")
    print(f"  USE_AMP        : {USE_AMP}")
    print(f"  GPU memory cap : {GPU_MEMORY_FRACTION * 100:.0f}%")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LR * 0.01,
    )

    if AMP_SUPPORTS_DEVICE_TYPE:
        scaler = GradScaler('cuda', enabled=USE_AMP)
    else:
        scaler = GradScaler(enabled=USE_AMP)

    if USE_MLFLOW:
        try:
            import mlflow
            mlflow.set_experiment("DS19_Two_Tower")
            mlflow.start_run(run_name=f"TwoTower_d{EMBED_DIM}_B{BATCH_SIZE}_4gbvram")
            mlflow.log_params({
                "model": "TwoTower",
                "embed_dim": EMBED_DIM,
                "num_heads": NUM_HEADS,
                "num_blocks": NUM_BLOCKS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "temperature": TEMPERATURE,
                "epochs": EPOCHS,
                "use_amp": USE_MLFLOW,
                "gpu_memory_fraction": GPU_MEMORY_FRACTION,
            })
            print("  ✅  MLflow tracking active")
        except Exception as e:
            print(f"  ⚠️  MLflow not available: {e}")
            USE_MLFLOW = False

    print_section("Training")
    print(f"  InfoNCE loss with temperature τ={TEMPERATURE}")
    print(f"  In-batch negatives: {BATCH_SIZE - 1} per positive")
    print(f"  Expected initial loss: ln({BATCH_SIZE}) = {math.log(BATCH_SIZE):.3f}")
    print()

    best_recall, best_epoch = train(
        model, train_loader, val_ds,
        optimizer, scaler, scheduler,
        device, n_items,
        start_epoch=0,
        best_recall=0.0,
    )

    final_path = MODELS_DIR / f"two_tower_epoch_{EPOCHS:03d}_final.pt"
    model_cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({'epoch': EPOCHS, 'model_state': model_cpu_state}, final_path)
    del model_cpu_state
    torch.cuda.empty_cache()

    print_section("Training Complete")
    print(f"  Best Sampled Recall@{EVAL_TOPK} : {best_recall:.4f} (epoch {best_epoch})")
    print(f"  Best model saved: models/saved/two_tower_best.pt")
    print(f"  Training log:     logs/two_tower_training_log.csv")

    if USE_MLFLOW:
        mlflow.log_metric("best_sampled_recall_50", best_recall)
        mlflow.log_artifact(str(LOGS_DIR / "two_tower_training_log.csv"))
        mlflow.end_run()

    print()
    print("  NEXT → Build FAISS index: python retrieval/faiss_index.py")

if __name__ == "__main__":
    main()