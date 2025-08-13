from sentence_transformers import SentenceTransformer
from typing import List

class Embedder:
    def __init__(self, model_name: str):
        # Loads a free, widely-used embedding model
        self.model = SentenceTransformer(model_name, device="cpu")
    def encode(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
