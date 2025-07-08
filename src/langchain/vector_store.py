from typing import List, Dict, Any
import numpy as np
import faiss

class VectorStore:
    def __init__(self, dimension: int = 384):
        """In-memory FAISS store using cosine similarity (inner-product on L2-normalized vectors)."""
        self.dimension = dimension
        # IndexFlatIP expects normalized vectors for cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        """Add L2-normalized embeddings & metadata to FAISS index."""
        # Normalize vectors to unit length for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        normalized = embeddings / norms
        self.index.add(normalized.astype('float32'))
        self.metadata.extend(metadatas)

    def search_similar(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Dict[str, Any]]:
        """Return top-`n_results` most similar items as list of dicts."""
        q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        D, I = self.index.search(q.reshape(1, -1).astype('float32'), n_results)
        results: List[Dict[str, Any]] = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({
                "score": float(score),
                **meta
            })
        return results

    def __len__(self):
        return len(self.metadata)