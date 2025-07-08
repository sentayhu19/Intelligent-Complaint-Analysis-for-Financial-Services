from typing import List, Dict, Any

import numpy as np

from src.langchain.embeddings import ComplaintEmbedder
from src.langchain.vector_store import VectorStore  # FAISS-based store implemented earlier


class Retriever:
    """Semantic retriever using shared embedder + FAISS vector store."""

    def __init__(self, store: VectorStore | None = None, top_k: int = 5):
        self.embedder = ComplaintEmbedder()
        self.store = store or VectorStore()
        self.top_k = top_k

    def retrieve(self, question: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Return top-k chunks sorted by similarity."""
        k = top_k or self.top_k
        q_emb = self.embedder.embed_texts([question])[0]
        results = self.store.search_similar(q_emb, n_results=k)
        return results

    def format_context(self, docs: List[Dict[str, Any]]) -> str:
        """Concatenate retrieved chunks into single context string."""
        parts = [f"Source {i+1}: {doc['metadata'].get('Complaint ID')}\n{doc['text']}" for i, doc in enumerate(docs)]
        return "\n\n".join(parts)
