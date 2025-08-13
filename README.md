# Customer Support RAG with Sentiment, Escalation & Empathy (Streamlit, 100% Free)

A fully local, free, Streamlit app that:
- Retrieves relevant help articles (RAG using Chroma + Sentence Transformers)
- Detects sentiment & emotion
- Predicts escalation risk (simple, transparent rules)
- Generates empathetic answers (FLAN-T5 small/base – free Hugging Face models)
- Tracks customer satisfaction (SQLite)

---

## 📜 Features
- **Retrieval-Augmented Generation (RAG)** – searches your help articles and answers using context from relevant documents.
- **Sentiment Analysis** – detects customer mood (positive, neutral, negative).
- **Emotion Detection** – identifies fine-grained emotions like anger, sadness, joy.
- **Escalation Prediction** – flags conversations likely to need human intervention.
- **Empathetic Response Generation** – adapts tone based on customer sentiment & emotion.
- **Customer Satisfaction Feedback** – thumbs up/down stored in SQLite for improvement tracking.
- **Local & Free Models** – no API keys required, runs fully offline after first model download.

---

## Demo Data
A few Markdown help articles are included under `data/sample_kb/articles`. Replace with your own.

## Quickstart

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

> The first run will download free Hugging Face models (embeddings, sentiment, emotion, generator).

## Project Structure

```
customer-support-rag-streamlit/
├─ app.py                      # Streamlit UI + chat loop
├─ requirements.txt
├─ README.md
├─ data/
│  └─ sample_kb/
│     └─ articles/            # Example Markdown articles
├─ src/
│  ├─ config.py               # Tunables (models, paths, thresholds, etc.)
│  ├─ data_ingest.py          # Load + chunk articles
│  ├─ embedder.py             # SentenceTransformers wrapper
│  ├─ vectorstore.py          # Chroma vector DB helpers
│  ├─ sentiment.py            # Sentiment + emotion detection
│  ├─ escalation.py           # Simple escalation scoring
│  ├─ generator.py            # FLAN-T5 text generation with empathy
│  ├─ rag_pipeline.py         # Glue: retrieve + generate
│  └─ utils.py                # Misc helpers (prompting, feedback store)
└─ storage/
   └─ chroma/                 # Local Chroma persistence
```

## Notes
- Everything uses free models and runs locally. For CPU-only, prefer `flan-t5-small`; for better quality, pick `flan-t5-base` in the sidebar.
- Replace the sample articles with your actual knowledge base (Markdown or plain text recommended).
- Satisfaction feedback is stored in `satisfaction.db` next to `app.py` (SQLite).
- Login with your huggingface token before running app.py

## License
MIT
