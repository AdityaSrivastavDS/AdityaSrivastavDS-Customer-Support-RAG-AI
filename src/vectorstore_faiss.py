import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int = 384):  # default for sentence-transformers/all-MiniLM-L6-v2
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product (cosine-like if normalized)
        self.docs = []

    def add_docs(self, docs, embeddings):
        # Convert to numpy float32
        embs = np.array(embeddings, dtype=np.float32)
        # Normalize for cosine similarity
        faiss.normalize_L2(embs)
        self.index.add(embs)
        self.docs.extend(docs)

    def query(self, query_emb, top_k: int = 5):
        q = np.array([query_emb], dtype=np.float32)
        faiss.normalize_L2(q)
        D, I = self.index.search(q, top_k)

        results = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            doc = self.docs[idx]
            results.append({
                "text": doc["text"],
                "source": doc.get("source", "unknown"),
                "score": float(score)
            })
        return results
