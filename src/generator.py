from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import List

class EmpatheticGenerator:
    def __init__(self, model_name: str = "google/flan-t5-small", max_new_tokens: int = 256, temperature: float = 0.3):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _tone_preamble(self, sentiment_label: str, emotion_label: str) -> str:
        if sentiment_label == "negative" or emotion_label in {"anger", "annoyance", "sadness", "fear"}:
            return "Use an empathetic, calm tone. Start with a brief apology and assurance. Keep sentences short and clear."
        elif sentiment_label == "positive":
            return "Use a warm, encouraging tone. Celebrate the progress. Keep it concise and helpful."
        else:
            return "Use a neutral, professional tone. Be clear and concise."

    def build_prompt(self, user_query: str, retrieved: List[dict], sentiment_label: str, emotion_label: str) -> str:
        tone = self._tone_preamble(sentiment_label, emotion_label)
        context = "\n\n".join([f"[Source {i+1}] {r['text']}\n(Source path: {r['source']})" for i, r in enumerate(retrieved)])
        instruction = f"""You are a helpful customer support assistant.
{tone}

Answer the customer's question strictly using the provided sources. 
If the information is missing, say you don't have that info and suggest next steps.
Cite sources as [Source N] inline when relevant.
Avoid making up policies or numbers.
"""
        prompt = f"""{instruction}

Customer question: {user_query}

Knowledge Base Context:
{context}

Final Answer:
"""
        return prompt

    def generate(self, prompt: str) -> str:
        out = self.pipe(prompt, max_new_tokens=self.max_new_tokens, do_sample=(self.temperature>0), temperature=self.temperature)
        return out[0]["generated_text"].strip()
