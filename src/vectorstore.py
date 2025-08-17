import sys
import sqlite3

import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, chroma_dir=None, collection_name="kb_collection"):
        # Use in-memory mode (safe for Streamlit Cloud)
        self.client = chromadb.Client(
            Settings(
                anonymized_telemetry=False,
                is_persistent=False
            )
        )
        # Make sure we always pass a valid name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_docs(self, docs, embeddings):
        """
        Adds document chunks with their embeddings into ChromaDB.
        docs: list of dicts with "id", "text", "source"
        embeddings: list of embedding vectors
        """
        ids = [str(i) for i in range(len(docs))]
        metadatas = [{"source": d["source"]} for d in docs]
        texts = [d["text"] for d in docs]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )

    def query(self, query_embedding, top_k=4):
        """
        Retrieves top_k most relevant chunks from ChromaDB.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return [
            {
                "text": doc,
                "source": meta.get("source", ""),
                "score": score
            }
            for doc, meta, score in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
