import time
import torch
import numpy as np
from typing import Tuple
from loguru import logger

from backend.app.core.config import TOP_K_RETRIEVAL, DEVICE
from backend.app.core.model_loader import ModelLoader


class RetrievalService:
    """
    Two-Tower retrieval: user sequence → user embedding → FAISS → top-100 candidates.
    """

    def __init__(self, model_loader: ModelLoader):
        self.ml            = model_loader
        self.device        = model_loader.device
        self.top_k_retrieve = TOP_K_RETRIEVAL
        self.item_idx_to_faiss_pos = {
            int(item_idx): pos
            for pos, item_idx in enumerate(self.ml.faiss_item_map.tolist())
        }

    @torch.no_grad()
    def retrieve(
        self,
        sequence_tensor: torch.Tensor,  # shape: [1, MAX_SEQ_LEN]
        top_k: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full retrieval pipeline: sequence → embedding → FAISS candidates.

        Args:
            sequence_tensor: padded user sequence [1, MAX_SEQ_LEN] (from FeatureService)
            top_k: number of candidates to retrieve (default: TOP_K_RETRIEVAL=100)

        Returns:
            candidate_item_idxs: np.ndarray [top_k] — internal item indices
            faiss_scores:        np.ndarray [top_k] — cosine similarity scores

        Steps:
            1. Move tensor to GPU
            2. Forward pass through User Tower only (not Item Tower)
            3. Get user embedding [1, 64]
            4. L2 normalize (FAISS uses inner product = cosine for normalized vecs)
            5. FAISS.search → (scores [1, top_k], positions [1, top_k])
            6. Map FAISS positions → item_idxs via faiss_item_map
        """
        if top_k is None:
            top_k = self.top_k_retrieve

        t0 = time.time()

        # ── Step 1: Move to device ───────────────────────────
        seq = sequence_tensor.to(self.device)  # [1, MAX_SEQ_LEN]

        # ── Step 2: User Tower forward pass ─────────────────
        # Two-Tower has encode_user(seq) method that returns the user embedding
        user_emb = self.ml.two_tower_model.encode_user(seq)  # [1, embed_dim]

        # ── Step 3: Move to CPU for FAISS ───────────────────
        user_emb_np = user_emb.cpu().float().numpy()  # [1, embed_dim]

        # Guard against non-finite outputs from the encoder (NaN/Inf can make FAISS return only -1 labels).
        if not np.isfinite(user_emb_np).all():
            bad_count = int(np.size(user_emb_np) - np.isfinite(user_emb_np).sum())
            logger.warning(
                f"Retrieval: user embedding had {bad_count} non-finite values; replacing with zeros"
            )
            user_emb_np = np.nan_to_num(user_emb_np, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Step 4: L2 normalize ────────────────────────────
        # FAISS IndexFlatIP does inner product; with L2-normalized vectors
        # this is equivalent to cosine similarity
        norm = np.linalg.norm(user_emb_np, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)   # avoid division by zero
        user_emb_norm = user_emb_np / norm   # [1, embed_dim]

        # Ensure float32 (FAISS requirement)
        user_emb_norm = user_emb_norm.astype(np.float32)

        # ── Step 5: FAISS search ─────────────────────────────
        scores, faiss_positions = self.ml.faiss_index.search(
            user_emb_norm, top_k
        )
        # scores:          [1, top_k] — inner product (cosine similarity)
        # faiss_positions: [1, top_k] — index positions in FAISS index

        scores         = scores[0]          # [top_k]
        faiss_positions = faiss_positions[0] # [top_k]

        # ── Step 6: Map positions → item_idxs ───────────────
        # faiss_item_map[position] = item_idx
        valid_mask = faiss_positions >= 0   # FAISS returns -1 for empty slots
        candidate_item_idxs = np.where(
            valid_mask,
            self.ml.faiss_item_map[np.clip(faiss_positions, 0, len(self.ml.faiss_item_map)-1)],
            -1
        )
        faiss_scores = np.where(valid_mask, scores, 0.0)

        # Filter out invalid (-1) results
        valid = candidate_item_idxs >= 0
        candidate_item_idxs = candidate_item_idxs[valid]
        faiss_scores        = faiss_scores[valid]

        retrieval_ms = (time.time() - t0) * 1000
        top_score = f"{faiss_scores[0]:.4f}" if len(faiss_scores) > 0 else "n/a"
        logger.debug(
            f"Retrieval: {len(candidate_item_idxs)} candidates "
            f"in {retrieval_ms:.1f}ms (top score: {top_score})"
        )

        return candidate_item_idxs, faiss_scores

    @torch.no_grad()
    def retrieve_similar_items(
        self,
        seed_item_idx: int,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Item-seeded retrieval using FAISS index vectors.

        Args:
            seed_item_idx: internal item index used as query seed
            top_k: number of similar items to return

        Returns:
            similar_item_idxs: np.ndarray [top_k]
            similarity_scores: np.ndarray [top_k]
        """
        if top_k <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        if seed_item_idx not in self.item_idx_to_faiss_pos:
            raise ValueError(f"seed_item_idx={seed_item_idx} not found in FAISS map")

        seed_pos = int(self.item_idx_to_faiss_pos[seed_item_idx])

        # Reconstruct the seed vector from FAISS and use it as the query.
        seed_vector = self.ml.faiss_index.reconstruct(seed_pos)
        query = np.asarray(seed_vector, dtype=np.float32).reshape(1, -1)

        norm = np.linalg.norm(query, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        query = (query / norm).astype(np.float32)

        # Request one extra candidate to account for the seed item itself.
        scores, positions = self.ml.faiss_index.search(query, top_k + 1)
        scores = scores[0]
        positions = positions[0]

        valid_mask = positions >= 0
        candidate_item_idxs = np.where(
            valid_mask,
            self.ml.faiss_item_map[np.clip(positions, 0, len(self.ml.faiss_item_map) - 1)],
            -1,
        )
        candidate_scores = np.where(valid_mask, scores, 0.0)

        out_idxs = []
        out_scores = []
        seen = set()
        for idx, score in zip(candidate_item_idxs, candidate_scores):
            item_idx = int(idx)
            if item_idx < 0 or item_idx == int(seed_item_idx) or item_idx in seen:
                continue
            seen.add(item_idx)
            out_idxs.append(item_idx)
            out_scores.append(float(score))
            if len(out_idxs) >= top_k:
                break

        return np.asarray(out_idxs, dtype=np.int64), np.asarray(out_scores, dtype=np.float32)

    def health_check(self) -> dict:
        """Check retrieval system status."""
        try:
            n_vectors = self.ml.faiss_index.ntotal
            model_ok  = self.ml.two_tower_model is not None
            return {
                "status": "healthy",
                "detail": (
                    f"{n_vectors:,} vectors indexed, "
                    f"model on {self.device}"
                )
            }
        except Exception as e:
            return {"status": "unavailable", "detail": str(e)}