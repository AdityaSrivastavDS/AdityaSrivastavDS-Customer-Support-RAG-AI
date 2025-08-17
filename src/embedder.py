from sentence_transformers import SentenceTransformer
from typing import List


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        # ✅ Store embedding dimension
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
