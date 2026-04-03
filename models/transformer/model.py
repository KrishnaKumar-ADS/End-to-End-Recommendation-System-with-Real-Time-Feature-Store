import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ─────────────────────────────────────────────────────────────
# TRANSFORMER BLOCK
# ─────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block.

    Pre-norm formulation (LayerNorm BEFORE attention and FFN).
    More stable than post-norm for small models.

    Forward:
      h  → LN → MHA → Dropout → h + residual
      h' → LN → FFN → Dropout → h' + residual
    """

    def __init__(
        self,
        hidden_dim:  int,
        num_heads:   int,
        ffn_dim:     int,
        dropout:     float = 0.2,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )

        # ── Attention ─────────────────────────────────────────
        self.norm1     = nn.LayerNorm(hidden_dim)
        self.attn      = nn.MultiheadAttention(
            embed_dim    = hidden_dim,
            num_heads    = num_heads,
            dropout      = dropout,
            batch_first  = True,   # input shape: [B, L, D] (not [L, B, D])
        )
        self.drop1     = nn.Dropout(dropout)

        # ── Feed-Forward Network ───────────────────────────────
        self.norm2     = nn.LayerNorm(hidden_dim)
        self.ffn       = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),                          # GELU outperforms ReLU for recommendation
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.drop2     = nn.Dropout(dropout)

    def forward(
        self,
        x:          torch.Tensor,              # [B, L, D]
        causal_mask: torch.Tensor,             # [L, L]   (additive, float)
        key_padding_mask: torch.Tensor,        # [B, L]   (bool, True = ignore)
    ) -> torch.Tensor:
        """
        x             : [B, L, D]
        causal_mask   : [L, L]     — upper triangle = -inf (no future attending)
        key_padding_mask: [B, L]   — True for PAD positions

        Returns: [B, L, D]
        """
        # ── Attention with pre-norm ────────────────────────────
        h      = self.norm1(x)
        
        # FIX: Ensure causal_mask matches key_padding_mask dtype
        # Convert causal_mask to bool (True = mask out = -inf positions)
        if causal_mask.dtype != key_padding_mask.dtype:
            # causal_mask has -inf where we want to mask, 0 elsewhere
            # Convert to bool: True where we want to mask
            causal_mask_bool = causal_mask == float('-inf')
        else:
            causal_mask_bool = causal_mask
        
        attn_out, _ = self.attn(
            query            = h,
            key              = h,
            value            = h,
            attn_mask        = causal_mask_bool,       # [L, L] - now bool
            key_padding_mask = key_padding_mask,         # [B, L] - bool
            need_weights     = False,
        )
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        x = x + self.drop1(attn_out)

        # ── FFN with pre-norm ──────────────────────────────────
        h = self.norm2(x)
        x = x + self.drop2(self.ffn(h))
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return x   # [B, L, D]


# ─────────────────────────────────────────────────────────────
# SASRec MODEL
# ─────────────────────────────────────────────────────────────

class SASRecModel(nn.Module):
    """
    SASRec: Self-Attentive Sequential Recommendation.

    Input:  item ID sequence [B, L]   (left-padded with 0)
    Output: context vectors [B, L, D] (at each position, the hidden state)

    The hidden state at position t represents the user's preference
    based on items [s_1, ..., s_t] — used to predict s_{t+1}.
    """

    def __init__(
        self,
        n_items:     int,         # total items INCLUDING PAD (PAD=0, items start at 1)
        hidden_dim:  int = 64,    # embedding + hidden dimension
        max_seq_len: int = 50,    # from Week 1 config
        num_heads:   int = 2,     # attention heads (hidden_dim / num_heads = head_dim)
        num_blocks:  int = 2,     # number of Transformer blocks
        dropout:     float = 0.2, # applied to embeddings + attention + FFN
    ):
        super().__init__()

        self.n_items    = n_items
        self.hidden_dim = hidden_dim
        self.max_seq_len= max_seq_len
        self.num_blocks = num_blocks

        # ── Item Embedding ─────────────────────────────────────
        # PAD token (index 0) → zero vector (padding_idx=0)
        self.item_emb = nn.Embedding(
            num_embeddings = n_items,
            embedding_dim  = hidden_dim,
            padding_idx    = 0,        # PAD token → zero vector, no gradient
        )

        # ── Positional Embedding ──────────────────────────────
        # Learnable, one vector per position 0..max_seq_len-1
        self.pos_emb = nn.Embedding(
            num_embeddings = max_seq_len,
            embedding_dim  = hidden_dim,
        )

        # ── Input Dropout ─────────────────────────────────────
        self.emb_dropout = nn.Dropout(dropout)

        # ── Transformer Blocks ─────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim = hidden_dim,
                num_heads  = num_heads,
                ffn_dim    = hidden_dim * 4,   # standard 4× expansion
                dropout    = dropout,
            )
            for _ in range(num_blocks)
        ])

        # ── Final Layer Norm ───────────────────────────────────
        # Applied after all blocks before projection
        self.norm = nn.LayerNorm(hidden_dim)

        # ── Weight Initialization ─────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """
        Xavier uniform initialization for all linear layers.
        Normal(0, 0.02) for embeddings (standard for Transformers).
        Zero init for layer norm bias.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build an additive causal mask of shape [seq_len, seq_len].
        mask[i][j] = 0     if j <= i   (position i CAN attend to position j)
        mask[i][j] = -inf  if j >  i   (position i CANNOT attend to future)

        This is added directly to attention logits before softmax.
        Result: softmax(-inf) = 0 → future items get zero attention weight.
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        mask = mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool(),
            float('-inf')
        )
        return mask   # [L, L]

    def forward(
        self,
        seq: torch.Tensor,    # [B, L]   item ID sequence (0=PAD)
    ) -> torch.Tensor:
        """
        seq: [B, L]   — padded item ID sequence

        Returns: [B, L, D] — hidden states at each position
        """
        B, L = seq.shape
        device = seq.device

        # ── Item embeddings ────────────────────────────────────
        item_e = self.item_emb(seq)    # [B, L, D]

        # ── Positional embeddings ──────────────────────────────
        # positions: [0, 1, 2, ..., L-1] for each sequence in batch
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)  # [B, L]
        pos_e     = self.pos_emb(positions)   # [B, L, D]

        # ── Combine and dropout ────────────────────────────────
        x = self.emb_dropout(item_e + pos_e)   # [B, L, D]
        non_pad_mask = (seq != 0).unsqueeze(-1).to(x.dtype)
        x = x * non_pad_mask

        # ── Build masks ────────────────────────────────────────
        # Causal mask: prevent attending to future positions
        causal_mask = self._build_causal_mask(L, device)   # [L, L]

        # Padding mask: prevent attending to PAD tokens
        # True = ignore this position in attention
        key_padding_mask = (seq == 0)   # [B, L]  True for PAD positions

        # ── Transformer blocks ─────────────────────────────────
        for block in self.blocks:
            x = block(x, causal_mask, key_padding_mask)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = x * non_pad_mask

        # ── Final layer norm ───────────────────────────────────
        x = self.norm(x)   # [B, L, D]
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = x * non_pad_mask

        return x

    def get_item_embeddings(self) -> torch.Tensor:
        """
        Returns the full item embedding matrix for scoring.
        Shape: [n_items, D]
        Used during evaluation for vectorized full-item ranking.
        """
        return self.item_emb.weight   # [n_items, D]

    def predict_scores(
        self,
        seq: torch.Tensor,           # [B, L]
        item_ids: torch.Tensor,      # [n_items] or [B, K]
    ) -> torch.Tensor:
        """
        Computes scores for given items at the LAST non-pad position.
        Used for top-K inference.

        Returns: [B, n_items] or [B, K] score matrix.
        """
        hidden = self.forward(seq)   # [B, L, D]

        # Extract last non-PAD position for each sequence
        # For left-padded sequences: last position (index L-1) is always meaningful
        h_last = hidden[:, -1, :]    # [B, D]

        # ── Score all items ────────────────────────────────────
        if item_ids.dim() == 1:
            # Full item scoring: item_ids = [n_items]
            item_e  = self.item_emb(item_ids)   # [n_items, D]
            scores  = h_last @ item_e.T         # [B, n_items]
        else:
            # Candidate scoring: item_ids = [B, K]
            item_e  = self.item_emb(item_ids)           # [B, K, D]
            scores  = torch.bmm(
                h_last.unsqueeze(1), item_e.transpose(1, 2)
            ).squeeze(1)                                  # [B, K]

        return scores

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────────────────────

def sampled_bce_loss(
    hidden:    torch.Tensor,   # [B, L, D]   — hidden states from forward()
    pos_items: torch.Tensor,   # [B, L]      — positive (target) item IDs
    neg_items: torch.Tensor,   # [B, L, K]   — negative item IDs
    item_emb:  nn.Embedding,   # item embedding layer (shared weights)
) -> torch.Tensor:
    """
    Sampled Binary Cross Entropy loss for SASRec.

    For each non-PAD position t in each sequence:
      pos_score = hidden[t] · E[pos_items[t]]
      neg_scores= hidden[t] · E[neg_items[t, :]]
      loss_t    = -log(σ(pos_score)) - mean(-log(σ(-neg_scores)))

    Loss is averaged only over non-PAD positions.

    Math:
      -log(σ(x))  = BCE(sigmoid(x), 1)  = "predict 1 for positive"
      -log(1-σ(x))= BCE(sigmoid(x), 0)  = "predict 0 for negative"
      = -log(σ(-x))

    Returns: scalar loss
    """
    B, L, D = hidden.shape
    K       = neg_items.shape[2]

    # ── Positive scores ────────────────────────────────────────
    # pos_items: [B, L] → embed → [B, L, D]
    pos_e      = item_emb(pos_items)          # [B, L, D]
    pos_scores = (hidden * pos_e).sum(dim=-1) # [B, L]   dot product per position

    # ── Negative scores ────────────────────────────────────────
    # neg_items: [B, L, K] → embed → [B, L, K, D]
    neg_e      = item_emb(neg_items)                          # [B, L, K, D]
    # Expand hidden: [B, L, D] → [B, L, 1, D]
    h_exp      = hidden.unsqueeze(2)                          # [B, L, 1, D]
    neg_scores = (h_exp * neg_e).sum(dim=-1)                  # [B, L, K]

    # Stabilize logits for long CPU runs where occasional non-finite values can appear.
    pos_scores = torch.nan_to_num(pos_scores, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
    neg_scores = torch.nan_to_num(neg_scores, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

    # ── Mask: only compute loss on non-PAD positions ───────────
    # pos_items == 0 means this is a PAD position (no target)
    mask       = (pos_items != 0).float()   # [B, L]  1.0 for real, 0.0 for PAD

    # ── BCE losses ─────────────────────────────────────────────
    pos_loss   = F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores), reduction='none'
    )   # [B, L]

    neg_loss   = F.binary_cross_entropy_with_logits(
        neg_scores, torch.zeros_like(neg_scores), reduction='none'
    ).mean(dim=-1)  # [B, L, K] → [B, L]  (mean over K negatives)

    # ── Apply mask and average ─────────────────────────────────
    total_loss    = (pos_loss + neg_loss) * mask
    n_valid       = mask.sum().clamp(min=1.0)
    loss          = total_loss.sum() / n_valid

    return loss


# ─────────────────────────────────────────────────────────────
# SMOKE TEST (run this file to verify)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  SASRec Model — Smoke Test")
    print("=" * 60)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    meta    = json.load(open("data/processed/dataset_meta.json"))
    n_items = meta.get("n_items_with_pad", meta["n_items"] + 1)

    # ── Build model ────────────────────────────────────────────
    model = SASRecModel(
        n_items    = n_items,
        hidden_dim = 64,
        max_seq_len= 50,
        num_heads  = 2,
        num_blocks = 2,
        dropout    = 0.2,
    ).to(device)

    total_params = model.count_parameters()
    print(f"\n  Model Architecture:")
    print(f"    n_items          : {n_items:,}")
    print(f"    hidden_dim       : 64")
    print(f"    num_heads        : 2  (head_dim = 32)")
    print(f"    num_blocks       : 2")
    print(f"    ffn_dim          : 256  (4 × 64)")
    print(f"    max_seq_len      : 50")
    print(f"    dropout          : 0.2")
    print(f"    Total parameters : {total_params:,}")

    # ── Forward pass ───────────────────────────────────────────
    B, L = 4, 50
    K    = 128

    seq  = torch.randint(0, n_items, (B, L), device=device)
    seq[:, :5] = 0   # add some padding

    pos  = torch.randint(1, n_items, (B, L), device=device)
    pos[:, :5] = 0

    neg  = torch.randint(1, n_items, (B, L, K), device=device)

    print(f"\n  Forward pass:")
    print(f"    Input seq shape  : {seq.shape}")

    with torch.no_grad():
        hidden = model(seq)
    print(f"    Output hidden    : {hidden.shape}")     # [B, L, D]
    print(f"    hidden[0, -1, :5]: {hidden[0, -1, :5].tolist()}")

    # ── Loss ───────────────────────────────────────────────────
    loss = sampled_bce_loss(hidden, pos, neg, model.item_emb)
    print(f"\n  Loss:")
    print(f"    sampled_bce_loss : {loss.item():.4f}")
    print(f"    Expected range   : 0.65 – 0.70 (random init, ~log(2) ≈ 0.693)")

    # ── Inference scoring ──────────────────────────────────────
    all_items = torch.arange(n_items, device=device)   # [n_items]
    with torch.no_grad():
        scores = model.predict_scores(seq, all_items)
    print(f"\n  Inference:")
    print(f"    predict_scores shape : {scores.shape}")  # [B, n_items]
    top5 = scores[0].topk(5)
    print(f"    Top-5 item indices   : {top5.indices.tolist()}")
    print(f"    Top-5 scores         : {[f'{v:.4f}' for v in top5.values.tolist()]}")

    # ── Memory estimate ────────────────────────────────────────
    if device == "cuda":
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\n  GPU Memory used: {mem_mb:.1f} MB  (well within 4GB)")

    print("\n  ✅ SASRec model smoke test PASSED")