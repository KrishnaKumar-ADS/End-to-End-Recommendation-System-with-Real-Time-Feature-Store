"""
search package - hybrid code search pipeline.

Modules
-------
bm25_index      BM25 tokenizer and in-memory index builder.
bm25_store      Redis-backed index serialization and lifecycle management.
hybrid_fusion   Reciprocal Rank Fusion over vector + BM25 result lists.
reranker        Cross-encoder reranking using sentence-transformers.
query_expander  Qwen-powered query expansion.
search_service  End-to-end search orchestrator.
evaluator       MRR@10, NDCG@10, Precision@k evaluation helpers.
"""
