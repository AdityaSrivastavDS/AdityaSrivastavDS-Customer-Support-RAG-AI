import chromadb
from chromadb.config import Settings
from typing import List, Dict
from pathlib import Path

class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "kb_collection"):
        self.client = chromadb.PersistentClient(path=str(persist_dir), settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})

    def reset(self):
        self.client.reset()

    def add_docs(self, docs: List[Dict], embeddings):
        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        metadatas = [{"source": d["source"]} for d in docs]
        self.collection.add(documents=texts, ids=ids, metadatas=metadatas, embeddings=embeddings.tolist())

    def query(self, query_emb, top_k: int = 4) -> List[Dict]:
        res = self.collection.query(query_embeddings=[query_emb.tolist()], n_results=top_k)
        out = []
        for doc, meta, score, _id in zip(res["documents"][0], res["metadatas"][0], res["distances"][0], res["ids"][0]):
            out.append({"id": _id, "text": doc, "source": meta.get("source", ""), "score": 1.0 - score})
        return out
