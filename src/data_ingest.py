from pathlib import Path
from typing import List, Dict
import re

def read_text_files(folder: Path) -> Dict[str, str]:
    texts = {}
    for p in folder.glob("**/*"):
        if p.suffix.lower() in {".md", ".txt"} and p.is_file():
            texts[str(p)] = p.read_text(encoding="utf-8", errors="ignore")
    return texts

def simple_clean(text: str) -> str:
    # Minimal cleanup; keep it readable for display.
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, chunk_size: int = 450, overlap: int = 60) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def build_corpus(folder: Path, chunk_size: int, overlap: int) -> List[Dict]:
    docs = []
    files = read_text_files(folder)
    for path, text in files.items():
        text = simple_clean(text)
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            docs.append({
                "id": f"{path}-chunk-{i}",
                "text": chunk,
                "source": path
            })
    return docs
