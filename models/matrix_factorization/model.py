import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────
# MATRIX FACTORIZATION MODEL
# ─────────────────────────────────────────────────────────────

class MatrixFactorization(nn.Module):
    """
    Matrix Factorization recommender model with BPR loss.

    Args:
        n_users       : number of users
        n_items       : number of items (INCLUDING PAD at index 0)
        embedding_dim : dimension d of user and item embeddings
        dropout_p     : dropout probability applied to embeddings (default 0.0)

    Internal:
        user_emb : nn.Embedding of shape [n_users, embedding_dim]
        item_emb : nn.Embedding of shape [n_items, embedding_dim]
                   padding_idx=0 — PAD token gets zero embedding (won't update)
    """

    def __init__(
        self,
        n_users       : int,
        n_items       : int,
        embedding_dim : int   = 64,
        dropout_p     : float = 0.0,
    ):
        super().__init__()

        self.n_users       = n_users
        self.n_items       = n_items
        self.embedding_dim = embedding_dim

        # ── User Embedding ────────────────────────────────────
        # Shape: [n_users, embedding_dim]
        # Each user_idx maps to a d-dimensional vector
        self.user_emb = nn.Embedding(
            num_embeddings = n_users,
            embedding_dim  = embedding_dim,
        )

        # ── Item Embedding ────────────────────────────────────
        # Shape: [n_items, embedding_dim]
        # padding_idx=0: ensures the PAD token always has a zero embedding
        # and does not receive gradients during backprop
        self.item_emb = nn.Embedding(
            num_embeddings = n_items,
            embedding_dim  = embedding_dim,
            padding_idx    = 0,    # Index 0 = PAD, gets zero embedding
        )

        # ── Optional Dropout ─────────────────────────────────
        # Regularization: randomly zeros out embedding dimensions during training
        # Prevents co-adaptation of embedding dimensions
        self.dropout = nn.Dropout(p=dropout_p)

        # ── Initialize Embeddings ────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Initialize embeddings with small Gaussian noise.

        WHY N(0, 0.01)?
            - Starts embeddings close to zero
            - Dot products start near zero (balanced scores)
            - Gradients flow cleanly from the start
            - Avoids saturation of sigmoid in BPR loss
            - Standard initialization for all embedding models
        """
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)

        # Zero out PAD embedding explicitly (though padding_idx handles this)
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0.0)

    # ─────────────────────────────────────────────────────────
    # FORWARD: TRAINING (BPR TRIPLET INPUT)
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        user_idx  : torch.Tensor,   # shape: [B]
        pos_item  : torch.Tensor,   # shape: [B]
        neg_item  : torch.Tensor,   # shape: [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for TRAINING.
        Computes positive and negative scores for BPR loss.

        Args:
            user_idx : batch of user indices  [B]
            pos_item : batch of positive item indices [B]
            neg_item : batch of negative item indices [B]

        Returns:
            pos_scores : [B]  scores for positive pairs
            neg_scores : [B]  scores for negative pairs
            reg_loss   : scalar  L2 regularization term on embeddings

        Usage:
            pos_scores, neg_scores, reg = model(users, pos_items, neg_items)
            bpr_loss = bpr_loss_fn(pos_scores, neg_scores)
            total_loss = bpr_loss + lambda_reg * reg
        """
        # Lookup embeddings
        # user_emb_b: [B, d]
        # pos_emb_b : [B, d]
        # neg_emb_b : [B, d]
        user_emb_b = self.dropout(self.user_emb(user_idx))
        pos_emb_b  = self.dropout(self.item_emb(pos_item))
        neg_emb_b  = self.dropout(self.item_emb(neg_item))

        # ── Dot Product Scores ───────────────────────────────
        # (user_emb_b * pos_emb_b): element-wise multiply → [B, d]
        # .sum(dim=-1): sum over embedding dim → [B]
        # This is equivalent to: batch of dot products
        pos_scores = (user_emb_b * pos_emb_b).sum(dim=-1)   # [B]
        neg_scores = (user_emb_b * neg_emb_b).sum(dim=-1)   # [B]

        # ── L2 Regularization Term ───────────────────────────
        # Penalizes large embedding norms → prevents overfitting
        # We regularize only the embeddings in this batch (not all embeddings)
        # This is equivalent to L2 weight decay, but applied at the batch level
        reg_loss = (
            user_emb_b.norm(2).pow(2) +
            pos_emb_b.norm(2).pow(2) +
            neg_emb_b.norm(2).pow(2)
        ) / user_emb_b.shape[0]

        return pos_scores, neg_scores, reg_loss

    # ─────────────────────────────────────────────────────────
    # GET USER EMBEDDING
    # ─────────────────────────────────────────────────────────

    def get_user_embedding(self, user_idx: torch.Tensor) -> torch.Tensor:
        """
        Returns the embedding vector for given users.

        Args:
            user_idx: [B] or scalar int

        Returns:
            user_emb: [B, d] or [d]
        """
        return self.user_emb(user_idx)

    # ─────────────────────────────────────────────────────────
    # GET ITEM EMBEDDING
    # ─────────────────────────────────────────────────────────

    def get_item_embedding(self, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Returns the embedding vector for given items.

        Args:
            item_idx: [B] or scalar int

        Returns:
            item_emb: [B, d] or [d]
        """
        return self.item_emb(item_idx)

    # ─────────────────────────────────────────────────────────
    # SCORE: USER vs SINGLE ITEM
    # ─────────────────────────────────────────────────────────

    def score(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute score for (user, item) pairs.
        Used during inference and evaluation.

        Args:
            user_idx: [B]
            item_idx: [B]

        Returns:
            scores: [B]
        """
        user_emb = self.user_emb(user_idx)   # [B, d]
        item_emb = self.item_emb(item_idx)   # [B, d]
        return (user_emb * item_emb).sum(dim=-1)   # [B]

    # ─────────────────────────────────────────────────────────
    # SCORE: USER vs ALL ITEMS (FOR FULL RANKING)
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def score_all_items(self, user_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute scores for one user against ALL items.
        Used during evaluation to get full item ranking.

        This is:  scores[i] = E_user · E_item[i]  for all i
        Equivalent to:  scores = E_user @ E_items.T

        Args:
            user_idx: scalar tensor or [1]

        Returns:
            scores: [n_items]  (score for every item, indexed 0..n_items-1)
        """
        user_emb   = self.user_emb(user_idx)         # [d]
        all_items  = self.item_emb.weight             # [n_items, d]
        scores     = all_items @ user_emb.unsqueeze(-1)  # [n_items, 1]
        return scores.squeeze(-1)                    # [n_items]

    # ─────────────────────────────────────────────────────────
    # MODEL INFO
    # ─────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"MatrixFactorization(\n"
            f"  n_users={self.n_users:,}\n"
            f"  n_items={self.n_items:,}\n"
            f"  embedding_dim={self.embedding_dim}\n"
            f"  total_params={self.num_parameters():,}\n"
            f"  user_emb shape: {list(self.user_emb.weight.shape)}\n"
            f"  item_emb shape: {list(self.item_emb.weight.shape)}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────
# BPR LOSS FUNCTION
# ─────────────────────────────────────────────────────────────

def bpr_loss(
    pos_scores : torch.Tensor,    # [B]  scores for positive items
    neg_scores : torch.Tensor,    # [B]  scores for negative items
) -> torch.Tensor:
    """
    Bayesian Personalized Ranking (BPR) loss.

    BPR Loss = -mean(log(sigmoid(pos_score - neg_score)))

    Derivation:
        We want to maximize P(pos > neg) for each user.
        P(pos > neg) = σ(score(u,i) - score(u,j))
        We maximize log P → minimize -log P
        Loss = -mean(log(σ(score_pos - score_neg)))

    Gradient behavior:
        When pos_score >> neg_score → σ ≈ 1 → log ≈ 0 → small gradient (already correct)
        When pos_score ≈ neg_score  → σ ≈ 0.5 → large gradient (needs correction)
        When pos_score << neg_score → σ ≈ 0 → log → -∞ → large gradient (inverted)

    Args:
        pos_scores: scores for positive items [B]
        neg_scores: scores for negative items [B]

    Returns:
        loss: scalar tensor
    """
    # score difference: positive minus negative
    # We want this to be as large as possible
    diff = pos_scores - neg_scores    # [B]

    # Sigmoid of difference = P(positive ranks above negative)
    # log(sigmoid) = numerically stable via F.logsigmoid
    loss = -F.logsigmoid(diff).mean()

    return loss


# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  Matrix Factorization Model — Smoke Test")
    print("=" * 60)

    # ── Init model ───────────────────────────────────────────
    model = MatrixFactorization(
        n_users       = 100_000,
        n_items       = 60_001,   # 0=PAD, 1..60000=real items
        embedding_dim = 64,
        dropout_p     = 0.1,
    )

    print(f"\n  Model:\n{model}")
    print(f"\n  Parameters: {model.num_parameters():,}")
    print(f"    User emb : {100_000 * 64:,} params")
    print(f"    Item emb : {60_001 * 64:,} params")

    # ── Forward pass ─────────────────────────────────────────
    B = 32  # batch size
    users    = torch.randint(0, 100_000, (B,))
    pos_items = torch.randint(1, 60_001, (B,))
    neg_items = torch.randint(1, 60_001, (B,))

    model.train()
    pos_scores, neg_scores, reg = model(users, pos_items, neg_items)

    print(f"\n  Forward pass (batch_size={B}):")
    print(f"    pos_scores shape : {pos_scores.shape}")
    print(f"    neg_scores shape : {neg_scores.shape}")
    print(f"    reg_loss         : {reg.item():.4f}")
    print(f"    pos_scores[:5]   : {pos_scores[:5].detach().tolist()}")
    print(f"    neg_scores[:5]   : {neg_scores[:5].detach().tolist()}")

    # ── BPR Loss ─────────────────────────────────────────────
    loss = bpr_loss(pos_scores, neg_scores)
    print(f"\n  BPR Loss: {loss.item():.6f}")
    print(f"  Expected initial loss ≈ 0.693 (= -log(0.5), random embeddings)")

    # ── Backward pass ────────────────────────────────────────
    loss.backward()
    print(f"  ✅ Backward pass succeeded")

    # ── score_all_items ──────────────────────────────────────
    model.eval()
    user_tensor = torch.tensor(0)
    all_scores = model.score_all_items(user_tensor)
    print(f"\n  score_all_items(user=0):")
    print(f"    Output shape : {all_scores.shape}   (expected: [{model.n_items}])")
    print(f"    Score[0] (PAD) : {all_scores[0].item():.6f}  (should be ≈ 0)")
    print(f"    Score[1..5]    : {all_scores[1:6].tolist()}")

    # ── PAD embedding is zero ─────────────────────────────────
    pad_emb = model.item_emb.weight[0]
    print(f"\n  PAD embedding norm (should be 0): {pad_emb.norm().item():.6f}")
    assert pad_emb.norm().item() < 1e-6, "PAD embedding should be zero!"

    print(f"\n  ✅ Model smoke test PASSED")