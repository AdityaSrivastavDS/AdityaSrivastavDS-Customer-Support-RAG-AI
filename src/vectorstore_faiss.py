import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, collection_name="support_kb"):
        # In-memory client (bypasses sqlite issue on Streamlit Cloud)
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_docs(self, docs, embeddings):
        ids = [f"doc_{i}" for i in range(len(docs))]
        texts = [d["text"] for d in docs]
        metadatas = [{"source": d["source"]} for d in docs]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    def query(self, query_emb, top_k=4):
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        docs = []
        for i in range(len(results["ids"][0])):
            docs.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "score": results["distances"][0][i],
            })
        return docs
