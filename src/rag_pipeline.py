from typing import List, Dict
from .embedder import Embedder
from .vectorstore_faiss import VectorStore   # This will be the modified version (no sqlite)
from .generator import EmpatheticGenerator
from .sentiment import SentimentEmotion

class RAGPipeline:
    def __init__(self, embed_model: str, generator_model: str, sentiment_model: str, emotion_model: str, top_k: int):
        self.embedder = Embedder(embed_model)
        # ✅ Initialize without chroma_dir (in-memory store)
        self.store = VectorStore()
        self.generator = EmpatheticGenerator(generator_model)
        self.se = SentimentEmotion(sentiment_model, emotion_model)
        self.top_k = top_k

    def index(self, docs: List[Dict]):
        embeddings = self.embedder.encode([d["text"] for d in docs])
        self.store.add_docs(docs, embeddings)

    def retrieve(self, query: str) -> List[Dict]:
        q_emb = self.embedder.encode([query])[0]
        return self.store.query(q_emb, top_k=self.top_k)

    def analyze(self, text: str) -> Dict:
        return self.se.analyze(text)

    def answer(self, user_query: str, retrieved: List[Dict], sentiment_label: str, emotion_label: str) -> str:
        prompt = self.generator.build_prompt(user_query, retrieved, sentiment_label, emotion_label)
        return self.generator.generate(prompt)
