import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, collection_name: str = "kb_collection"):
        # In-memory Chroma client
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        
        # Create or get collection
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

    def query(self, query_embeddings, n_results: int):
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
