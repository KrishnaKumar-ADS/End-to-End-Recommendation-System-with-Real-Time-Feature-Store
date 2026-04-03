import os
import sys
import json
import time
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# PATH SETUP (Allow running from project root)
# ─────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.matrix_factorization.model import MatrixFactorization, bpr_loss
from models.matrix_factorization.dataset import (
    MFTrainDataset, MFEvalDataset,
    load_interaction_data, build_user_item_sets, build_dataloaders,
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

# Paths
PROCESSED_DIR  = Path("data/processed")
SPLITS_DIR     = Path("data/splits")
MODELS_DIR     = Path("models/saved")
LOGS_DIR       = Path("logs")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
EMBEDDING_DIM     = 64        # Size of user/item embedding vectors
BATCH_SIZE        = 1024      # Training batch size
LEARNING_RATE     = 1e-3      # Adam learning rate
WEIGHT_DECAY      = 1e-6      # L2 regularization coefficient (via Adam)
LAMBDA_REG        = 1e-6      # BPR regularization coefficient (manual)
EPOCHS            = 20        # Maximum training epochs
PATIENCE          = 5         # Early stopping patience (epochs without improvement)
DROPOUT_P         = 0.1       # Embedding dropout rate
NEG_STRATEGY      = 'uniform' # Negative sampling: 'uniform' or 'popularity'
EVAL_EVERY        = 1         # Evaluate every N epochs
K_VALUES          = [5, 10, 20]  # K values for HR@K and NDCG@K evaluation
SEED              = 42

# ─────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def print_section(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def get_device() -> torch.device:
    """Selects best available device: CUDA > MPS (Apple) > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  ✅ Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"  ✅ Device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"  ✅ Device: CPU")
    return device

# ─────────────────────────────────────────────────────────────
# TRAINING STEP
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model      : MatrixFactorization,
    loader     : DataLoader,
    optimizer  : torch.optim.Optimizer,
    device     : torch.device,
    lambda_reg : float,
    epoch      : int,
) -> Dict[str, float]:
    """
    Runs one full training epoch.

    For each batch:
      1. Load (user, pos_item, neg_item) triplets
      2. Forward pass → pos_scores, neg_scores
      3. Compute BPR loss + regularization loss
      4. Backward pass → update embeddings

    Args:
        model      : MatrixFactorization model
        loader     : DataLoader yielding (user, pos, neg) batches
        optimizer  : Adam optimizer
        device     : torch.device
        lambda_reg : weight for L2 regularization term
        epoch      : current epoch number (for tqdm display)

    Returns:
        metrics: dict with keys 'loss', 'bpr_loss', 'reg_loss'
    """
    model.train()

    total_loss     = 0.0
    total_bpr_loss = 0.0
    total_reg_loss = 0.0
    n_batches      = 0

    # tqdm progress bar
    pbar = tqdm(
        loader,
        desc    = f"  Epoch {epoch:03d} [Train]",
        unit    = "batch",
        ncols   = 90,
        leave   = True,
    )

    for batch_users, batch_pos, batch_neg in pbar:

        # Move to device
        batch_users = batch_users.to(device, non_blocking=True)
        batch_pos   = batch_pos.to(device, non_blocking=True)
        batch_neg   = batch_neg.to(device, non_blocking=True)

        # ── Forward pass ──────────────────────────────────────
        optimizer.zero_grad()
        pos_scores, neg_scores, reg = model(batch_users, batch_pos, batch_neg)

        # ── BPR Loss ──────────────────────────────────────────
        loss_bpr = bpr_loss(pos_scores, neg_scores)

        # ── Total Loss ────────────────────────────────────────
        # Combine BPR loss + L2 regularization
        # lambda_reg controls how much to penalize large embeddings
        loss = loss_bpr + lambda_reg * reg

        # ── Backward & Update ────────────────────────────────
        loss.backward()

        # Gradient clipping: prevents gradient explosion
        # Max norm = 1.0 is a safe default for embedding models
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ── Accumulate Metrics ────────────────────────────────
        total_loss     += loss.item()
        total_bpr_loss += loss_bpr.item()
        total_reg_loss += reg.item()
        n_batches      += 1

        # Update progress bar
        pbar.set_postfix({
            'loss'    : f"{loss.item():.4f}",
            'bpr'     : f"{loss_bpr.item():.4f}",
        })

    pbar.close()

    avg_loss     = total_loss     / n_batches
    avg_bpr_loss = total_bpr_loss / n_batches
    avg_reg_loss = total_reg_loss / n_batches

    return {
        'loss'     : avg_loss,
        'bpr_loss' : avg_bpr_loss,
        'reg_loss' : avg_reg_loss,
    }


# ─────────────────────────────────────────────────────────────
# EMBEDDING SCORE COMPUTATION (FOR EVALUATION)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_all_item_scores_batch(
    model    : MatrixFactorization,
    users    : torch.Tensor,   # [B]
    device   : torch.device,
) -> torch.Tensor:
    """
    Computes scores for each user in the batch against ALL items.

    This vectorized approach avoids looping over items.

    Shape:  users [B], output [B, n_items]

    How:
      user_embs  = model.user_emb(users)     → [B, d]
      all_items  = model.item_emb.weight     → [n_items, d]
      scores     = user_embs @ all_items.T   → [B, n_items]
    """
    model.eval()
    users      = users.to(device)
    user_embs  = model.user_emb(users)         # [B, d]
    all_items  = model.item_emb.weight         # [n_items, d]
    scores     = user_embs @ all_items.t()     # [B, n_items]
    return scores                              # [B, n_items]


# ─────────────────────────────────────────────────────────────
# EVALUATION STEP
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model      : MatrixFactorization,
    loader     : DataLoader,
    device     : torch.device,
    k_values   : List[int],
    user_items : Dict[int, set],
    n_items    : int,
    split_name : str = "val",
) -> Dict[str, float]:
    """
    Evaluates the model using full-item ranking.

    For each user:
      1. Compute scores against ALL items
      2. Mask out items the user has already interacted with (training history)
         → This ensures we don't recommend items the user already consumed
      3. Rank remaining items by score (descending)
      4. Check if ground truth item is in Top-K
      5. Compute HR@K and NDCG@K

    IMPORTANT — The masking step:
        Without masking, the model might "recommend" items from the user's history,
        which trivially score very high (they were trained as positives).
        We must exclude these to get a fair evaluation.

    Args:
        model      : trained MatrixFactorization model
        loader     : DataLoader yielding (user_idx, gt_item) pairs
        device     : torch.device
        k_values   : list of K values to evaluate, e.g. [5, 10, 20]
        user_items : {user_idx: set of ALL interacted items} for masking
        n_items    : number of items (index range: 0..n_items)
        split_name : 'val' or 'test' (for printing)

    Returns:
        metrics: {
            'HR@5': 0.62, 'HR@10': 0.71, 'HR@20': 0.80,
            'NDCG@5': 0.45, 'NDCG@10': 0.48, 'NDCG@20': 0.50,
            'MRR@10': 0.39
        }
    """
    model.eval()

    # Accumulators for each K
    hits     = {k: 0.0 for k in k_values}
    ndcgs    = {k: 0.0 for k in k_values}
    mrr_vals = []
    n_users  = 0

    pbar = tqdm(
        loader,
        desc  = f"  [{split_name}] Evaluating",
        unit  = "batch",
        ncols = 90,
        leave = True,
    )

    for batch_users, batch_gt in pbar:
        B = batch_users.shape[0]

        # Compute scores: [B, n_items]
        scores = compute_all_item_scores_batch(model, batch_users, device)

        # Process each user in batch
        for i in range(B):
            user_idx = batch_users[i].item()
            gt_item  = batch_gt[i].item()
            user_scores = scores[i].cpu().numpy().copy()  # [n_items]

            # ── Mask Training History ────────────────────────
            # Set scores of already-seen items to -inf
            # This prevents "recommending" items the user already consumed
            seen_items = user_items.get(user_idx, set())
            for seen in seen_items:
                if 0 <= seen < len(user_scores):
                    user_scores[seen] = -np.inf

            # ── Also mask PAD token ──────────────────────────
            user_scores[0] = -np.inf

            # ── Rank ALL items ───────────────────────────────
            # argsort in DESCENDING order
            ranked_items = np.argsort(-user_scores)  # [n_items]

            # Find rank of ground truth item (1-indexed)
            # np.where returns a tuple; [0][0] gives the first match position
            gt_positions = np.where(ranked_items == gt_item)[0]
            if len(gt_positions) == 0:
                # GT item not found — should not happen
                n_users += 1
                continue
            rank = int(gt_positions[0]) + 1   # Convert to 1-indexed rank

            # ── HR@K: Did the ground truth appear in top K? ──
            for k in k_values:
                if rank <= k:
                    hits[k] += 1.0

            # ── NDCG@K: Discounted rank reward ──────────────
            # NDCG@K = 1 / log2(rank + 1)  if rank <= K, else 0
            for k in k_values:
                if rank <= k:
                    ndcgs[k] += 1.0 / np.log2(rank + 1)

            # ── MRR@K (using K=10 by default) ───────────────
            max_k = max(k_values)
            if rank <= max_k:
                mrr_vals.append(1.0 / rank)
            else:
                mrr_vals.append(0.0)

            n_users += 1

    pbar.close()

    # Compute averages
    metrics = {}
    for k in k_values:
        metrics[f'HR@{k}']   = hits[k]  / n_users
        metrics[f'NDCG@{k}'] = ndcgs[k] / n_users

    metrics[f'MRR@{max(k_values)}'] = float(np.mean(mrr_vals))
    metrics['n_users_evaluated']     = n_users

    return metrics


# ─────────────────────────────────────────────────────────────
# CHECKPOINT MANAGEMENT
# ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model       : MatrixFactorization,
    optimizer   : torch.optim.Optimizer,
    epoch       : int,
    metrics     : dict,
    filename    : str,
):
    """
    Saves a model checkpoint to disk.

    Checkpoint contains:
        - model state_dict  (learned embeddings)
        - optimizer state_dict (Adam momentum, etc.)
        - epoch number
        - evaluation metrics at this checkpoint
        - model hyperparameters (for reconstruction)

    Args:
        model    : MatrixFactorization model
        optimizer: optimizer instance
        epoch    : current epoch
        metrics  : evaluation metrics dict
        filename : path to save (e.g. 'models/saved/mf_best.pt')
    """
    checkpoint = {
        'epoch'        : epoch,
        'model_state'  : model.state_dict(),
        'optim_state'  : optimizer.state_dict(),
        'metrics'      : metrics,
        'hyperparams'  : {
            'n_users'       : model.n_users,
            'n_items'       : model.n_items,
            'embedding_dim' : model.embedding_dim,
        },
    }
    torch.save(checkpoint, filename)
    print(f"\n  💾 Checkpoint saved: {filename}")


def load_checkpoint(
    model     : MatrixFactorization,
    optimizer : Optional[torch.optim.Optimizer],
    filename  : str,
    device    : torch.device,
) -> Tuple[int, dict]:
    """
    Loads a model checkpoint from disk.

    Returns:
        epoch   : epoch the checkpoint was saved at
        metrics : evaluation metrics at that checkpoint
    """
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optim_state'])
    print(f"  ✅ Loaded checkpoint from: {filename}")
    print(f"     Epoch   : {checkpoint['epoch']}")
    print(f"     Metrics : {checkpoint['metrics']}")
    return checkpoint['epoch'], checkpoint['metrics']


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

class TrainingLogger:
    """
    Logs training metrics to both console and a CSV file.

    Each row in the CSV = one epoch.
    Columns: epoch, train_loss, HR@5, HR@10, HR@20, NDCG@5, NDCG@10, NDCG@20, elapsed_sec
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.rows     = []
        print(f"\n  Training log will be saved to: {log_path}")

    def log(self, epoch: int, train_metrics: dict, eval_metrics: dict, elapsed: float):
        row = {
            'epoch'      : epoch,
            'train_loss' : train_metrics.get('loss', 0),
            'bpr_loss'   : train_metrics.get('bpr_loss', 0),
            'elapsed_sec': elapsed,
        }
        row.update(eval_metrics)
        self.rows.append(row)

        # Print to console
        print(
            f"\n  Epoch {epoch:03d} | "
            f"loss={train_metrics.get('loss', 0):.4f} | "
            f"HR@10={eval_metrics.get('HR@10', 0):.4f} | "
            f"NDCG@10={eval_metrics.get('NDCG@10', 0):.4f} | "
            f"MRR@20={eval_metrics.get('MRR@20', 0):.4f} | "
            f"time={elapsed:.1f}s"
        )

    def save(self):
        df = pd.DataFrame(self.rows)
        df.to_csv(self.log_path, index=False)
        print(f"\n  💾 Training log saved: {self.log_path}")


# ─────────────────────────────────────────────────────────────
# MLFLOW TRACKING
# ─────────────────────────────────────────────────────────────

def setup_mlflow(experiment_name: str = "DS19_MatrixFactorization"):
    """
    Sets up MLflow experiment tracking.

    MLflow tracks:
      - Hyperparameters (embedding_dim, lr, batch_size, etc.)
      - Per-epoch metrics (loss, HR@10, NDCG@10)
      - Model artifacts (saved checkpoint)

    If MLflow is not installed or the server is not running,
    this gracefully returns None and training continues without tracking.
    """
    try:
        import mlflow
        mlflow.set_experiment(experiment_name)
        print(f"  ✅ MLflow experiment: {experiment_name}")
        return mlflow
    except ImportError:
        print("  ⚠️  MLflow not installed. Skipping experiment tracking.")
        print("     Install with: pip install mlflow")
        return None
    except Exception as e:
        print(f"  ⚠️  MLflow setup failed: {e}")
        print("     Training will continue without MLflow.")
        return None


# ─────────────────────────────────────────────────────────────
# EARLY STOPPING
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation metric stops improving.

    Monitors HR@10 (or any specified metric).
    If no improvement for `patience` consecutive epochs → stop.

    Args:
        patience : number of epochs to wait for improvement
        metric   : metric name to monitor (higher = better)
        min_delta: minimum change to count as improvement
    """

    def __init__(self, patience: int = 5, metric: str = 'HR@10', min_delta: float = 1e-4):
        self.patience  = patience
        self.metric    = metric
        self.min_delta = min_delta
        self.best_val  = -np.inf
        self.wait      = 0
        self.stopped   = False

    def step(self, current_val: float) -> bool:
        """
        Returns True if training should STOP.
        """
        if current_val > self.best_val + self.min_delta:
            self.best_val = current_val
            self.wait     = 0
        else:
            self.wait += 1
            print(f"  ⚠️  No improvement in {self.metric} for {self.wait}/{self.patience} epochs")

        if self.wait >= self.patience:
            self.stopped = True
            print(f"  🛑 Early stopping triggered after {self.patience} epochs without improvement.")
            return True

        return False


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────
def sanitize(name):
    """Sanitize metric name for MLflow compatibility."""
    return name.replace('@', '_at_')


def main():
    print_section("DS19 — Week 2 | Matrix Factorization Training")
    set_seed(SEED)

    # ── Device ────────────────────────────────────────────────
    print_section("Device Setup")
    device = get_device()

    # ── Load Data ─────────────────────────────────────────────
    df, meta, train_seqs, val_labels, test_labels = load_interaction_data()

    with open(SPLITS_DIR / "val_seqs.pkl", "rb") as f:
        val_seqs = pickle.load(f)

    n_users = meta['n_users']
    n_items_with_pad = meta.get('n_items_with_pad', meta['n_items'] + 1)
    n_items = n_items_with_pad - 1  # real items = 1..n_items

    # ── Build User-Item Sets ──────────────────────────────────
    user_items = build_user_item_sets(train_seqs)

    # ── Build Datasets ────────────────────────────────────────
    print_section("Building Datasets")

    train_dataset = MFTrainDataset(
        train_seqs  = train_seqs,
        user_items  = user_items,
        n_items     = n_items,
        neg_per_pos = 1,
        strategy    = NEG_STRATEGY,
    )

    val_dataset = MFEvalDataset(
        labels     = val_labels,
        input_seqs = val_seqs,
    )

    train_loader, val_loader = build_dataloaders(
        train_dataset, val_dataset,
        batch_size  = BATCH_SIZE,
        num_workers = 0,
    )

    # ── Build Model ───────────────────────────────────────────
    print_section("Building Model")

    model = MatrixFactorization(
        n_users       = n_users,
        n_items       = n_items_with_pad,
        embedding_dim = EMBEDDING_DIM,
        dropout_p     = DROPOUT_P,
    ).to(device)

    print(f"\n  {model}")

    # ── Optimizer & Scheduler ─────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,   # L2 via Adam
    )

    # ReduceLROnPlateau: halve LR when HR@10 stops improving for 3 epochs
    # This is standard practice: start with 1e-3, decay to 1e-4, 1e-5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = 'max',      # we want to maximize HR@10
        factor   = 0.5,        # new_lr = old_lr * 0.5
        patience = 3,          # wait 3 epochs before reducing
        verbose  = True,
    )

    # ── Early Stopping ────────────────────────────────────────
    early_stopper = EarlyStopping(patience=PATIENCE, metric='HR@10')

    # ── MLflow ────────────────────────────────────────────────
    mlflow = setup_mlflow("DS19_MatrixFactorization")

    # ── Logger ────────────────────────────────────────────────
    log_path = LOGS_DIR / "mf_training_log.csv"
    logger   = TrainingLogger(log_path)

    # ── Training Config Summary ───────────────────────────────
    print_section("Training Configuration")
    print(f"  n_users        : {n_users:,}")
    print(f"  n_items        : {n_items:,}  (+ 1 PAD = {n_items_with_pad:,})")
    print(f"  embedding_dim  : {EMBEDDING_DIM}")
    print(f"  batch_size     : {BATCH_SIZE}")
    print(f"  learning_rate  : {LEARNING_RATE}")
    print(f"  weight_decay   : {WEIGHT_DECAY}")
    print(f"  lambda_reg     : {LAMBDA_REG}")
    print(f"  epochs         : {EPOCHS}")
    print(f"  patience       : {PATIENCE}")
    print(f"  neg_strategy   : {NEG_STRATEGY}")
    print(f"  k_values       : {K_VALUES}")
    print(f"  total_params   : {model.num_parameters():,}")
    print(f"  train_batches  : {len(train_loader):,}")
    print(f"  val_users      : {len(val_dataset):,}")

    # ── Start MLflow Run ──────────────────────────────────────
    if mlflow:
        mlflow.start_run(run_name=f"MF_dim{EMBEDDING_DIM}_lr{LEARNING_RATE}")
        mlflow.log_params({
            'embedding_dim'  : EMBEDDING_DIM,
            'batch_size'     : BATCH_SIZE,
            'learning_rate'  : LEARNING_RATE,
            'weight_decay'   : WEIGHT_DECAY,
            'lambda_reg'     : LAMBDA_REG,
            'epochs'         : EPOCHS,
            'neg_strategy'   : NEG_STRATEGY,
            'n_users'        : n_users,
            'n_items'        : n_items,
            'dropout_p'      : DROPOUT_P,
        })

    # ─────────────────────────────────────────────────────────
    # TRAINING LOOP
    # ─────────────────────────────────────────────────────────

    print_section("Training Loop")

    best_hr10    = -np.inf
    best_epoch   = -1
    best_metrics = {}
    run_start    = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # ── 1. Train for one epoch ─────────────────────────────
        train_metrics = train_one_epoch(
            model      = model,
            loader     = train_loader,
            optimizer  = optimizer,
            device     = device,
            lambda_reg = LAMBDA_REG,
            epoch      = epoch,
        )

        # ── 2. Evaluate on validation set ─────────────────────
        if epoch % EVAL_EVERY == 0:
            eval_metrics = evaluate(
                model      = model,
                loader     = val_loader,
                device     = device,
                k_values   = K_VALUES,
                user_items = user_items,
                n_items    = n_items_with_pad,
                split_name = "val",
            )
        else:
            eval_metrics = {}

        # ── 3. Learning rate scheduling ────────────────────────
        current_hr10 = eval_metrics.get('HR@10', 0.0)
        if current_hr10 > 0:
            scheduler.step(current_hr10)

        # ── 4. Log epoch results ───────────────────────────────
        elapsed = time.time() - epoch_start
        logger.log(epoch, train_metrics, eval_metrics, elapsed)

        if mlflow:
            # Sanitize metric names for MLflow compatibility
            mlflow_metrics = {
                'train_loss' : train_metrics.get('loss', 0),
                'bpr_loss'   : train_metrics.get('bpr_loss', 0),
            }
            # Add eval metrics with sanitized names
            for k, v in eval_metrics.items():
                if isinstance(v, float):
                    sanitized_key = sanitize(k)
                    mlflow_metrics[sanitized_key] = v
            
            mlflow.log_metrics(mlflow_metrics, step=epoch)

        # ── 5. Save best checkpoint ────────────────────────────
        if current_hr10 > best_hr10:
            best_hr10    = current_hr10
            best_epoch   = epoch
            best_metrics = eval_metrics.copy()

            save_checkpoint(
                model     = model,
                optimizer = optimizer,
                epoch     = epoch,
                metrics   = eval_metrics,
                filename  = str(MODELS_DIR / "mf_best.pt"),
            )
            print(f"  ⭐ New best HR@10: {best_hr10:.4f}  (epoch {epoch})")

        # ── 6. Save periodic checkpoint every 5 epochs ────────
        if epoch % 5 == 0:
            save_checkpoint(
                model     = model,
                optimizer = optimizer,
                epoch     = epoch,
                metrics   = eval_metrics,
                filename  = str(MODELS_DIR / f"mf_epoch_{epoch:03d}.pt"),
            )

        # ── 7. Early stopping check ────────────────────────────
        if current_hr10 > 0:
            should_stop = early_stopper.step(current_hr10)
            if should_stop:
                print(f"\n  🛑 Early stopping at epoch {epoch}")
                break

    # ─────────────────────────────────────────────────────────
    # TRAINING COMPLETE
    # ─────────────────────────────────────────────────────────

    total_time = time.time() - run_start
    logger.save()

    print_section("Training Complete")
    print(f"\n  Total training time : {total_time/60:.1f} minutes")
    print(f"  Best epoch          : {best_epoch}")
    print(f"  Best HR@10          : {best_hr10:.4f}")
    print(f"\n  Best Validation Metrics:")
    for k, v in best_metrics.items():
        if isinstance(v, float):
            print(f"    {k:<20} : {v:.4f}")

    # ── End MLflow Run ────────────────────────────────────────
    if mlflow:
        mlflow.log_metrics({
            'best_HR_at_10'   : best_hr10,
            'best_epoch'   : float(best_epoch),
            'total_time_s' : total_time,
        })
        mlflow.log_artifact(str(MODELS_DIR / "mf_best.pt"))
        mlflow.end_run()
        print(f"\n  ✅ MLflow run complete")

    # ── Final summary ─────────────────────────────────────────
    print()
    print("=" * 60)
    print("✅  Week 2 — Matrix Factorization Training Complete!")
    print("=" * 60)
    print(f"\n  Outputs:")
    print(f"    models/saved/mf_best.pt             ← Best model checkpoint")
    print(f"    models/saved/mf_epoch_*.pt          ← Periodic checkpoints")
    print(f"    logs/mf_training_log.csv            ← Per-epoch metrics")
    print(f"\n  Next: python experiments/evaluation.py  ← Full test-set evaluation")
    print("=" * 60)


if __name__ == "__main__":
    main()