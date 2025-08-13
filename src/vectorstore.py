import sys
import pysqlite3  # Needed only if you still want SQLite locally

# Patch for systems with old sqlite3
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, chroma_dir=None, collection_name="kb_collection"):
        # Force in-memory mode for Streamlit Cloud
        self.client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                is_persistent=False  # 🚀 No SQLite, runs entirely in RAM
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, embeddings, metadatas, documents):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )

    def query(self, query_embedding, top_k=4):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
