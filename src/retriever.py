# src/retriever.py

import numpy as np
from sentence_transformers import SentenceTransformer
import json
import faiss


MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------
# Load embedding model
# ---------------------------
def load_model():
    return SentenceTransformer(MODEL_NAME)


# ---------------------------
# Load FAISS index + metadata
# ---------------------------
def load_index(index_dir, name):

    index = faiss.read_index(f"{index_dir}/{name}.faiss")

    with open(f"{index_dir}/{name}_meta.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    chunk_ids = list(metadata.keys())

    return index, metadata, chunk_ids


# ---------------------------
# Convert L2 distance → cosine-like score
# ---------------------------
def l2_to_cosine(dist):
    return 1 / (1 + dist)


# ---------------------------
# MMR selection
# ---------------------------
def mmr(query_emb, doc_embs, top_k=5, lambda_param=0.8):

    selected = []
    candidates = list(range(len(doc_embs)))

    sim_to_query = doc_embs @ query_emb.T

    while len(selected) < top_k and candidates:

        if not selected:
            idx = np.argmax(sim_to_query)
            selected.append(idx)
            candidates.remove(idx)
            continue

        mmr_scores = []

        for c in candidates:

            relevance = sim_to_query[c]

            diversity = max(
                doc_embs[c] @ doc_embs[s].T
                for s in selected
            )

            score = lambda_param * relevance - (1 - lambda_param) * diversity

            mmr_scores.append((score, c))

        idx = max(mmr_scores)[1]

        selected.append(idx)

        candidates.remove(idx)

    return selected


# ---------------------------
# Retrieval
# ---------------------------
def retrieve(query,
             index,
             metadata,
             chunk_ids,
             model,
             k=7,
             doc_type=None,
             threshold=0.35):

    # Encode query
    q_emb = model.encode([query], convert_to_numpy=True)[0]

    # Search FAISS
    D, I = index.search(np.array([q_emb]), k * 3)

    candidates = []

    for dist, idx in zip(D[0], I[0]):

        chunk_id = chunk_ids[idx]
        meta = metadata[chunk_id]

        if doc_type and meta["doc_type"] != doc_type:
            continue

        score = l2_to_cosine(dist)

        candidates.append({
            "chunk_id": chunk_id,
            "text": meta["text"],
            "source": meta["source"],
            "page": meta["page"],
            "doc_type": meta["doc_type"],
            "score": score
        })

    if not candidates:
        return [{
            "text": "Not enough information found",
            "score": 0
        }]

    # Create embeddings for MMR
    doc_embs = model.encode(
        [c["text"] for c in candidates],
        convert_to_numpy=True
    )

    selected_idx = mmr(
        q_emb,
        doc_embs,
        top_k=min(k, len(candidates))
    )

    results = [candidates[i] for i in selected_idx]

    if results[0]["score"] < threshold:
        return [{
            "text": "Not enough information found",
            "score": results[0]["score"]
        }]

    return results


if __name__ == "__main__":

    model = load_model()

    index, metadata, chunk_ids = load_index(
    "../output/faiss_indexes",
    "combined_index"
)

    query = "What is contract liability?"

    results = retrieve(query, index, metadata, chunk_ids, model)

    for r in results:
        print(r)