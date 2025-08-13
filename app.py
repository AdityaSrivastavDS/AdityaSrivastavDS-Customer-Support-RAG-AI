import streamlit as st
from pathlib import Path
import os, zipfile
import gdown

from src import config
from src.data_ingest import build_corpus
from src.rag_pipeline import RAGPipeline
from src.escalation import escalation_risk
from src.utils import init_sqlite, store_feedback

st.set_page_config(page_title="Customer Support RAG (Free)", page_icon="💬", layout="wide")

# ---------- Download Knowledge Base from Google Drive ----------
FILE_ID = "1_y2mlSEtbZRgWP6GSpZ37joeWPYguJUG"
KB_ZIP_URL = f"https://drive.google.com/uc?id={FILE_ID}"
os.makedirs(config.DATA_DIR, exist_ok=True)
zip_path = config.DATA_DIR / "kb.zip"

if not any(config.DATA_DIR.glob("*")):  # Download only if empty
    st.sidebar.write("📥 Downloading KB from Google Drive...")
    gdown.download(KB_ZIP_URL, str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(config.DATA_DIR)
    os.remove(zip_path)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
top_k = st.sidebar.slider("Top-K retrieval", 2, 8, config.TOP_K)
gen_model_choice = st.sidebar.selectbox(
    "Generator model",
    ["t5-small (CPU friendly)", "t5-base (better quality)"]
)
gen_model = config.GENERATOR_SMALL if "small" in gen_model_choice else config.GENERATOR_BASE

st.sidebar.markdown("---")
st.sidebar.caption("All models are free Hugging Face models downloaded on first run.")

# ---------- Initialize DB for feedback ----------
init_sqlite(config.SQLITE_DB)

# ---------- Session State ----------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "history" not in st.session_state:
    st.session_state.history = []
if "neg_streak" not in st.session_state:
    st.session_state.neg_streak = 0
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

# ---------- Build / Load Index ----------
col1, col2 = st.columns([1, 2])
with col1:
    st.header("📚 Knowledge Base")
    if st.button("(Re)Index KB"):
        folder = Path(config.DATA_DIR)
        st.write("Loading and chunking documents...")
        corpus = build_corpus(folder, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        st.write(f"Found {len(corpus)} chunks. Creating embeddings and indexing...")

        st.session_state.pipeline = RAGPipeline(
            embed_model=config.EMBED_MODEL,
            generator_model=gen_model,
            sentiment_model=config.SENTIMENT_MODEL,
            emotion_model=config.EMOTION_MODEL,
            chroma_dir=None,  # In-memory
            top_k=top_k
        )
        st.session_state.pipeline.index(corpus)
        st.success("Index ready ✅")

with col2:
    st.header("💬 Customer Support Assistant")

    if st.session_state.pipeline is None:
        st.info("Press **(Re)Index KB** to initialize the system with your knowledge base.")
    else:
        user_input = st.text_input(
            "Customer message",
            placeholder="e.g., I was charged after I cancelled. This is really frustrating."
        )
        ask = st.button("Ask")

        if ask and user_input.strip():
            analysis = st.session_state.pipeline.analyze(user_input)
            sentiment = analysis["sentiment_label"]
            emotion = analysis["emotion_label"]

            if sentiment == "negative":
                st.session_state.neg_streak += 1
            else:
                st.session_state.neg_streak = 0

            retrieved = st.session_state.pipeline.retrieve(user_input)
            answer = st.session_state.pipeline.answer(
                user_input, retrieved, sentiment, emotion
            )
            st.session_state.last_answer = answer

            esc = escalation_risk(
                sentiment_label=sentiment,
                emotion_label=emotion,
                neg_streak=st.session_state.neg_streak,
                threshold=config.ESCALATION_SCORE_THRESHOLD,
                neg_streak_escalate=config.NEGATIVE_STREAK_ESCALATE
            )

            with st.container():
                st.markdown("### 🧠 AI Answer")
                st.write(answer)

                st.markdown("#### 🔎 Sources")
                for i, r in enumerate(retrieved, start=1):
                    st.markdown(f"**Source {i}** (score {r['score']:.2f}) — `{r['source']}`")
                    st.caption(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))

                st.markdown("#### 🧭 Understanding the Customer")
                c1, c2, c3 = st.columns(3)
                c1.metric("Sentiment", sentiment.capitalize(), delta=f"{analysis['sentiment_score']:.2f}")
                c2.metric("Emotion", emotion.capitalize(), delta=f"{analysis['emotion_score']:.2f}")
                c3.metric("Escalation Risk", f"{esc.risk_score:.2f}", delta=esc.reason)
                if esc.should_escalate:
                    st.warning("🚩 Escalation suggested — please consider routing to a human agent.")

            st.session_state.history.append({"role": "user", "text": user_input, "sentiment_label": sentiment})
            st.session_state.history.append({"role": "assistant", "text": answer, "sentiment_label": "n/a"})

        st.markdown("### 🗂 Conversation")
        for turn in st.session_state.history[-8:]:
            if turn["role"] == "user":
                st.chat_message("user").write(turn["text"])
            else:
                st.chat_message("assistant").write(turn["text"])

        if st.session_state.last_answer:
            col_ok, col_bad = st.columns(2)
            if col_ok.button("Resolved 👍"):
                store_feedback(config.SQLITE_DB, st.session_state.history[-2]["text"], st.session_state.last_answer, True)
                st.success("Thanks! Marked as resolved.")
            if col_bad.button("Not helpful 👎"):
                store_feedback(config.SQLITE_DB, st.session_state.history[-2]["text"], st.session_state.last_answer, False)
                st.info("Feedback noted. We'll try to improve.")

st.markdown("---")
st.caption("100% free local stack • Chroma + Sentence Transformers + Transformers (T5) • Built for empathetic support.")
