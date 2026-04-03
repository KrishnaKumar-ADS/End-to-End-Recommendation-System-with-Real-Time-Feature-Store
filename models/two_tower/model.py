import sys
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")

EMBED_DIM     = 64
NUM_HEADS     = 2
NUM_BLOCKS    = 2
DROPOUT       = 0.2
MAX_SEQ_LEN   = 50
PAD_TOKEN     = 0
TEMPERATURE   = 0.07

# ─────────────────────────────────────────────────────────────
# TRANSFORMER BLOCK (same as SASRec — reusable)
# ─────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer encoder block (SASRec-style).

    Structure:
      LayerNorm → MultiHeadAttention → Dropout → Residual
      LayerNorm → FFN(d → 4d → d) → Dropout → Residual

    Pre-norm (LN before sub-layer) is more stable than post-norm
    for small hidden dimensions (d=64).
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()

        self.norm1   = nn.LayerNorm(hidden_dim)
        self.norm2   = nn.LayerNorm(hidden_dim)

        self.attn    = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,       # input: [B, L, d]
        )

        self.ffn     = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor = None,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [B, L, d]
        attn_mask: [L, L] causal mask (True = blocked)
        key_padding_mask: [B, L] True where sequence is PAD
        Returns: [B, L, d]
        """

        # ── Self-attention with pre-norm ──────────────────────
        h = self.norm1(x)
        h, _ = self.attn(
            query             = h,
            key               = h,
            value             = h,
            attn_mask         = attn_mask,
            key_padding_mask  = key_padding_mask,
            need_weights      = False,
        )
        x = x + self.dropout(h)

        # ── FFN with pre-norm ─────────────────────────────────
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)

        return x

# ─────────────────────────────────────────────────────────────
# USER TOWER
# ─────────────────────────────────────────────────────────────

class UserTower(nn.Module):
    """
    Encodes a user's interaction sequence into a dense representation.

    Input:  seq ∈ Z^{B × L}  (batch of padded item sequences)
    Output: h_u ∈ R^{B × d}  (L2-normalized user embeddings)

    Architecture:
      1. Item embedding lookup:   x_l = E_items[i_l]
      2. Positional embedding:    x_l += E_pos[l]
      3. Dropout on input
      4. N × TransformerBlock with causal mask + padding mask
      5. Extract last non-padding position
      6. Linear projection: d → d
      7. L2 normalize

    Shared item embedding matrix is passed in (not owned by this module).
    """

    def __init__(self,
                 item_embedding: nn.Embedding,
                 hidden_dim: int     = EMBED_DIM,
                 num_heads: int      = NUM_HEADS,
                 num_blocks: int     = NUM_BLOCKS,
                 max_seq_len: int    = MAX_SEQ_LEN,
                 dropout: float      = DROPOUT):
        super().__init__()

        self.item_embedding = item_embedding          # shared with ItemTower
        self.pos_embedding  = nn.Embedding(max_seq_len, hidden_dim)
        self.input_dropout  = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        self.norm       = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, hidden_dim)

        self.hidden_dim  = hidden_dim
        self.max_seq_len = max_seq_len
        self._pad_token  = PAD_TOKEN

        # ── Causal mask (upper triangle = True = masked) ──────
        # Shape: [L, L]
        # attn_mask[i, j] = True → position i CANNOT attend to j
        # We want causal: position i can attend only to 1..i
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, L] — padded item sequences (PAD=0)
        Returns: [B, d] — L2-normalized user embeddings
        """
        B, L = seq.shape

        # ── Item embeddings ───────────────────────────────────
        x = self.item_embedding(seq)    # [B, L, d]

        # ── Positional embeddings ─────────────────────────────
        # positions: 0, 1, ..., L-1
        positions = torch.arange(L, device=seq.device).unsqueeze(0)  # [1, L]
        x = x + self.pos_embedding(positions)                         # [B, L, d]

        x = self.input_dropout(x)

        # ── Padding mask: True where token is PAD ─────────────
        # key_padding_mask: [B, L], True = ignore this position
        key_padding_mask = (seq == self._pad_token)    # [B, L]

        # Keep padded positions zeroed before attention.
        # With causal + key padding masks, fully-masked query rows can otherwise produce NaNs.
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # ── Transformer blocks ────────────────────────────────
        for block in self.blocks:
            x = block(
                x,
                attn_mask        = self.causal_mask[:L, :L],
                key_padding_mask = key_padding_mask,
            )
            # Prevent non-finite values from padded queries from propagating into later blocks.
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        x = self.norm(x)    # [B, L, d]

        # Final safety clamp before selecting user representation.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Extract last non-padding token ────────────────────
        # Robustly pick the last non-padding position for each user.
        # If a sequence is fully padded, fallback to position 0.
        pos_idx = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        last_non_pad = torch.where(
            key_padding_mask,
            torch.zeros_like(pos_idx),
            pos_idx,
        ).max(dim=1).values
        h_u = x[torch.arange(B, device=seq.device), last_non_pad]   # [B, d]

        # ── Project and normalize ─────────────────────────────
        h_u = self.projection(h_u)     # [B, d]
        h_u = torch.nan_to_num(h_u, nan=0.0, posinf=0.0, neginf=0.0)
        h_u = F.normalize(h_u, dim=-1)  # [B, d], unit sphere

        return h_u

# ─────────────────────────────────────────────────────────────
# ITEM TOWER
# ─────────────────────────────────────────────────────────────

class ItemTower(nn.Module):
    """
    Encodes a single item_id into a dense representation.

    Input:  item_ids ∈ Z^{B}  (batch of item IDs)
    Output: h_v ∈ R^{B × d}  (L2-normalized item embeddings)

    Architecture:
      1. Item embedding lookup: h_v = E_items[item_id]
      2. Linear projection: d → d
      3. L2 normalize

    Shared item embedding matrix is passed in (same as UserTower).
    """

    def __init__(self,
                 item_embedding: nn.Embedding,
                 hidden_dim: int = EMBED_DIM):
        super().__init__()

        self.item_embedding = item_embedding     # shared with UserTower
        self.projection     = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        item_ids: [B] or [N] — item IDs to encode
        Returns: [B, d] or [N, d] — L2-normalized item embeddings
        """
        h_v = self.item_embedding(item_ids)   # [B, d]
        h_v = self.projection(h_v)             # [B, d]
        h_v = F.normalize(h_v, dim=-1)         # [B, d], unit sphere
        return h_v

# ─────────────────────────────────────────────────────────────
# TWO-TOWER MODEL (combines both towers)
# ─────────────────────────────────────────────────────────────

class TwoTowerModel(nn.Module):
    """
    Full Two-Tower retrieval model.

    Combines UserTower + ItemTower with a shared item embedding matrix.
    Exposes:
      - encode_user(seq)      → user_embedding
      - encode_item(item_ids) → item_embeddings
      - forward(seq, item_ids)  → (user_embs, item_embs) for loss computation
    """

    def __init__(self,
                 n_items: int,
                 hidden_dim: int     = EMBED_DIM,
                 num_heads: int      = NUM_HEADS,
                 num_blocks: int     = NUM_BLOCKS,
                 max_seq_len: int    = MAX_SEQ_LEN,
                 dropout: float      = DROPOUT,
                 pad_token: int      = PAD_TOKEN):
        super().__init__()

        self.n_items    = n_items
        self.hidden_dim = hidden_dim

        # ── Shared item embedding matrix ──────────────────────
        # n_items + 1 rows: index 0 = PAD_TOKEN
        self.item_embedding = nn.Embedding(
            num_embeddings = n_items + 1,
            embedding_dim  = hidden_dim,
            padding_idx    = pad_token,
        )

        # ── Towers ────────────────────────────────────────────
        self.user_tower = UserTower(
            item_embedding = self.item_embedding,
            hidden_dim     = hidden_dim,
            num_heads      = num_heads,
            num_blocks     = num_blocks,
            max_seq_len    = max_seq_len,
            dropout        = dropout,
        )

        self.item_tower = ItemTower(
            item_embedding = self.item_embedding,
            hidden_dim     = hidden_dim,
        )

        # ── Initialize weights ────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def encode_user(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: [B, L]
        Returns: [B, d] — L2-normalized user embeddings
        """
        return self.user_tower(seq)

    def encode_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        item_ids: [B] or [N]
        Returns: [B, d] or [N, d] — L2-normalized item embeddings
        """
        return self.item_tower(item_ids)

    def forward(self, seq: torch.Tensor,
                item_ids: torch.Tensor) -> tuple:
        """
        seq:      [B, L] — user sequences
        item_ids: [B]    — target items (one per user in batch)
        Returns: (user_embs [B, d], item_embs [B, d])
        """
        user_embs = self.encode_user(seq)       # [B, d]
        item_embs = self.encode_item(item_ids)  # [B, d]
        return user_embs, item_embs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ─────────────────────────────────────────────────────────────
# INFONCE LOSS
# ─────────────────────────────────────────────────────────────

def infonce_loss(user_embs: torch.Tensor,
                 item_embs: torch.Tensor,
                 temperature: float = TEMPERATURE,
                 symmetric: bool = False) -> torch.Tensor:
    """
    InfoNCE loss with in-batch negatives.

    user_embs: [B, d]  — L2-normalized user embeddings
    item_embs: [B, d]  — L2-normalized item embeddings
    temperature: τ scalar

    Loss:
      S = user_embs @ item_embs.T / τ   ∈ R^{B × B}
      y = [0, 1, 2, ..., B-1]           ← diagonal is correct pair
      L = CrossEntropy(S, y)

    For symmetric loss (CLIP-style):
      L = (CrossEntropy(S, y) + CrossEntropy(S.T, y)) / 2
    """
    B = user_embs.size(0)

    # Similarity matrix: [B, B]
    # Since both are L2-normalized, this is cosine similarity
    S = torch.matmul(user_embs, item_embs.T) / temperature   # [B, B]

    # Labels: diagonal entries are positives
    labels = torch.arange(B, device=user_embs.device)

    if symmetric:
        loss_u2v = F.cross_entropy(S,   labels)
        loss_v2u = F.cross_entropy(S.T, labels)
        return (loss_u2v + loss_v2u) / 2.0
    else:
        return F.cross_entropy(S, labels)

# ─────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  Testing TwoTowerModel")
    print("=" * 60)

    # Load metadata
    meta    = json.load(open(PROCESSED_DIR / "dataset_meta.json"))
    n_items = meta['n_items']

    model = TwoTowerModel(
        n_items    = n_items,
        hidden_dim = EMBED_DIM,
        num_heads  = NUM_HEADS,
        num_blocks = NUM_BLOCKS,
        max_seq_len= MAX_SEQ_LEN,
        dropout    = DROPOUT,
    )

    n_params = model.count_parameters()
    print(f"  n_items       : {n_items:,}")
    print(f"  Parameters    : {n_params:,}  ({n_params/1e6:.2f}M)")

    # Test forward pass
    B = 8
    seq      = torch.randint(1, n_items + 1, (B, MAX_SEQ_LEN))
    seq[:, :10] = PAD_TOKEN   # Simulate left-padding
    item_ids = torch.randint(1, n_items + 1, (B,))

    user_embs, item_embs = model(seq, item_ids)

    print(f"  user_embs     : {user_embs.shape}")    # [8, 64]
    print(f"  item_embs     : {item_embs.shape}")    # [8, 64]

    # Check L2 normalization
    user_norms = user_embs.norm(dim=-1)
    item_norms = item_embs.norm(dim=-1)
    assert torch.allclose(user_norms, torch.ones(B), atol=1e-5), "User embs not L2 normalized"
    assert torch.allclose(item_norms, torch.ones(B), atol=1e-5), "Item embs not L2 normalized"
    print(f"  L2 norms      : user={user_norms.mean():.4f}, item={item_norms.mean():.4f}")

    # Test loss
    loss = infonce_loss(user_embs, item_embs, temperature=TEMPERATURE)
    print(f"  InfoNCE Loss  : {loss.item():.4f}  (should be close to ln(B)={math.log(B):.4f} at init)")

    # Test encode_item (for FAISS index building)
    all_items = torch.arange(1, 50, dtype=torch.long)  # small subset
    with torch.no_grad():
        all_embs = model.encode_item(all_items)
    print(f"  encode_item   : {all_embs.shape}")   # [49, 64]

    print("\n  ✅  Model tests passed.")
    print(f"  ✅  Expected VRAM: ~{n_params*4/1e6*2:.0f}MB (fp16 during training)")
    print("  ✅  Proceed to: python models/two_tower/train.py")