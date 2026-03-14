# src/embedder.py

"""
Embedding & FAISS Index Module
------------------------------
• Load chunked documents (chunks_500.json)
• SentenceTransformer embeddings (all-MiniLM-L6-v2)
• FAISS IndexFlatL2 (exact search)
• Separate domain indexes (legal / medical / combined)
• Save + Load index + metadata
• Benchmark embedding performance
• Saves to: output/faiss_indexes
"""

import json
import time
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

# Project root (AI-Powered RAG System)
BASE_DIR = Path(__file__).resolve().parent.parent

# Chunk file
CHUNKS_PATH = BASE_DIR / "output" / "chunks_500.json"

# FAISS output dir (REQUIRED PATH)
INDEX_DIR = BASE_DIR / "output" / "faiss_indexes"


# =========================================================
# LOAD CHUNKS
# =========================================================
def load_chunks(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Chunk file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if len(chunks) == 0:
        raise ValueError("[ERROR] Chunk file is empty")

    print(f"[INFO] Loaded chunks: {len(chunks)}")
    return chunks


# =========================================================
# LOAD MODEL
# =========================================================
def load_model():
    print("[INFO] Loading embedding model...")
    return SentenceTransformer(MODEL_NAME)


# =========================================================
# EMBEDDING
# =========================================================
def embed_chunks(chunks: List[Dict], model, batch_size=32) -> np.ndarray:
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    if embeddings.ndim != 2:
        raise ValueError("[ERROR] Embeddings must be 2D")

    return embeddings


# =========================================================
# BUILD FAISS
# =========================================================
def build_faiss_index(embeddings: np.ndarray):
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(embeddings)
    return index


# =========================================================
# SAVE INDEX + METADATA
# =========================================================
def save_index(index, chunks: List[Dict], name: str):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / f"{name}.faiss"
    meta_path = INDEX_DIR / f"{name}_meta.json"

    # Save FAISS
    faiss.write_index(index, str(index_path))

    # Save metadata
    metadata = {c["chunk_id"]: c for c in chunks}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SAVED] {index_path}")
    print(f"[SAVED] {meta_path}")


# =========================================================
# BUILD DOMAIN INDEXES
# =========================================================
def build_domain_indexes(chunks: List[Dict]):
    model = load_model()

    legal_chunks = [c for c in chunks if c["doc_type"] == "legal"]
    medical_chunks = [c for c in chunks if c["doc_type"] == "medical"]

    print(f"[INFO] Legal chunks: {len(legal_chunks)}")
    print(f"[INFO] Medical chunks: {len(medical_chunks)}")
    print(f"[INFO] Total chunks: {len(chunks)}")

    # LEGAL
    if legal_chunks:
        emb_legal = embed_chunks(legal_chunks, model)
        idx_legal = build_faiss_index(emb_legal)
        save_index(idx_legal, legal_chunks, "legal_index")

    # MEDICAL
    if medical_chunks:
        emb_med = embed_chunks(medical_chunks, model)
        idx_med = build_faiss_index(emb_med)
        save_index(idx_med, medical_chunks, "medical_index")

    # COMBINED
    emb_all = embed_chunks(chunks, model)
    idx_all = build_faiss_index(emb_all)
    save_index(idx_all, chunks, "combined_index")


# =========================================================
# LOAD INDEX
# =========================================================
def load_index(name: str):
    index_path = INDEX_DIR / f"{name}.faiss"
    meta_path = INDEX_DIR / f"{name}_meta.json"

    if not index_path.exists():
        raise FileNotFoundError(f"[ERROR] Index not found: {index_path}")

    index = faiss.read_index(str(index_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


# =========================================================
# BENCHMARK
# =========================================================
def benchmark_embedding(chunks: List[Dict], n=100):
    model = load_model()
    sample = chunks[:n]

    print(f"[INFO] Benchmarking {len(sample)} chunks...")

    start = time.time()
    emb = embed_chunks(sample, model)
    end = time.time()

    mem_mb = emb.nbytes / (1024 * 1024)

    print(f"[RESULT] Time: {end-start:.3f} sec")
    print(f"[RESULT] Shape: {emb.shape}")
    print(f"[RESULT] Memory: {mem_mb:.2f} MB")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    try:
        chunks = load_chunks(CHUNKS_PATH)

        build_domain_indexes(chunks)

        benchmark_embedding(chunks, 100)

        print("[DONE] Embedding & FAISS indexing complete")

    except Exception as e:
        print(e)