"""
retriever.py
Builds a lightweight in-memory vector store from stock data chunks
and retrieves the most relevant chunks for a given user query.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    Simple in-memory vector store using cosine similarity.
    No external DB needed — perfect for a portfolio-sized dataset.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print("  🔧 Loading embedding model...")
        self.model   = SentenceTransformer(model_name)
        self.chunks  : list[str]        = []
        self.vectors : np.ndarray | None = None
        print("  ✅ Embedding model ready.\n")

    def add_documents(self, portfolio_data: list[dict]) -> None:
        """Embed all chunks from portfolio_data and store them."""
        all_chunks = []
        for stock in portfolio_data:
            all_chunks.extend(stock["chunks"])

        print(f"  📚 Embedding {len(all_chunks)} knowledge chunks...")
        self.chunks  = all_chunks
        self.vectors = self.model.encode(all_chunks, show_progress_bar=False,
                                         convert_to_numpy=True)
        print(f"  ✅ Vector store built ({len(all_chunks)} chunks indexed).\n")

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return the top-k most relevant chunks for a query."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True)

        # Cosine similarity
        norms   = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10
        q_norm  = np.linalg.norm(query_vec) + 1e-10
        sims    = (self.vectors / norms) @ (query_vec / q_norm).T
        scores  = sims.flatten()

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.chunks[i] for i in top_idx]
