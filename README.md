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

## 🚀 How to Use

Once the project is launched (either locally or deployed), follow these steps:

### 1️⃣ Select Your Knowledge Base
- Click the **"(Re)Index KB"** button.
  - The system will load and chunk your documents.
  - It will then create **embeddings** and build a **vector database** for quick search.

---

### 2️⃣ Ask Questions
- In the **"Customer Support Assistant"** section, type your query in the text box.
- Click **"Ask"**.
- The AI will:
  1. **Analyze** your query (Sentiment + Emotion detection).
  2. **Retrieve** the most relevant context from your KB.
  3. **Generate** a personalized, empathetic answer.

---

### 3️⃣ View AI’s Response
- The AI’s generated answer will appear in the **"🧠 AI Answer"** section.
- Below it, you will find:
  - **Sources** used for the answer.
  - **Customer Understanding Metrics**:
    - Sentiment (Positive, Neutral, Negative)
    - Emotion (e.g., Anger, Joy, Sadness)
    - Escalation Risk (if the conversation should be passed to a human agent).

---

### 4️⃣ Provide Feedback
- At the bottom of each conversation, you can mark:
  - ✅ **Resolved** — if the AI’s answer was helpful.
  - ❌ **Not helpful** — if the answer needs improvement.
- Feedback will be stored in the local **SQLite database** (`satisfaction.db`).

---

### 5️⃣ Conversation History
- The latest **8 turns** of the conversation are shown.
- Older history is stored in memory for the current session.

---

### 💡 Notes
- **All models** used are free and downloaded from Hugging Face on first run.
- You can switch between **generator models** (`t5-small` for speed or `t5-base` for better quality) from the sidebar.
- No internet connection is required after models are downloaded (fully local processing).
