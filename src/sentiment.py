from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class SentimentEmotion:
    def __init__(self, sentiment_model: str, emotion_model: str):
        self.sentiment_pipe = pipeline("sentiment-analysis", model=sentiment_model)
        self.emotion_pipe = pipeline("text-classification", model=emotion_model, top_k=None)

    def analyze(self, text: str):
        sent = self.sentiment_pipe(text)[0]  # label: POSITIVE/NEGATIVE/NEUTRAL
        emotions = self.emotion_pipe(text)[0]  # list of dicts with 'label' and 'score'
        # Find the top emotion
        top_emotion = max(emotions, key=lambda x: x['score']) if emotions else {"label": "neutral", "score": 0.0}
        return {
            "sentiment_label": sent["label"].lower(),
            "sentiment_score": float(sent["score"]),
            "emotion_label": top_emotion["label"].lower(),
            "emotion_score": float(top_emotion["score"]),
            "emotions": emotions
        }
