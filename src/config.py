from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "knowledge_base"       # This will be downloaded at runtime
CHROMA_DIR = None                            # Using in-memory Chroma
SQLITE_DB = BASE_DIR / "satisfaction.db"

# Models (all free & public)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTIMENT_MODEL = "finiteautomata/bertweet-base-sentiment-analysis"
EMOTION_MODEL = "bhadresh-savani/distilbert-base-uncased-emotion"
GENERATOR_SMALL = "t5-small"
GENERATOR_BASE = "t5-base"

# RAG / Retrieval
CHUNK_SIZE = 450
CHUNK_OVERLAP = 60
TOP_K = 4

# Escalation thresholds
NEGATIVE_STREAK_ESCALATE = 2
ESCALATION_SCORE_THRESHOLD = 0.6

# Generation
MAX_TOKENS = 256
TEMPERATURE = 0.3
