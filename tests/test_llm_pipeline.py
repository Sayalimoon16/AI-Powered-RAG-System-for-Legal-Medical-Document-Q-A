import sys
import os
import json
import logging
from pathlib import Path

# -----------------------------
# Add project root to path
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


# -----------------------------
# Utils import (NEW)
# -----------------------------
from src.utils import setup_logger, validate_env, detect_language


# -----------------------------
# Setup logging + env check
# -----------------------------
setup_logger()
validate_env()


# -----------------------------
# Imports
# -----------------------------
from src.retriever import load_model, load_index, retrieve
from src.llm_chain import generate_answer


# -----------------------------
# Load embedding model
# -----------------------------
model = load_model()


# -----------------------------
# Load FAISS index
# -----------------------------
index_path = BASE_DIR / "outputs" / "faiss_indexes"

index, metadata, chunk_ids = load_index(
    index_path,
    "combined_index"
)


# -----------------------------
# Sample queries
# -----------------------------
queries = [
    "What is a contract agreement?",
    "Define liability clause",
    "What does the contract act say about agreements?",
    "What is the patient's glucose level?",
    "What treatment was provided?",
    "What follow-up instructions were given?",
    "What diagnosis was mentioned?",
    "Explain partnership definition",
    "What blood test values are reported?",
    "Is the patient medically fit?"
]


results_list = []


# -----------------------------
# Run queries
# -----------------------------
for q in queries:

    print("\n==============================")
    print("QUESTION:", q)

    try:

        # Language check (NEW)
        if not detect_language(q):
            print("Query must be in English.")
            continue


        results = retrieve(
            q,
            index,
            metadata,
            chunk_ids,
            model,
            k=5
        )


        answer = generate_answer(q, results)

        print("\nANSWER:\n", answer)


        sources = []

        if isinstance(results, list) and results != ["Not enough information found"]:

            for r in results:

                sources.append({
                    "file": r["source"],
                    "page": r["page"],
                    "score": float(r["score"])
                })


        results_list.append({
            "question": q,
            "answer": answer,
            "sources": sources
        })


    except Exception as e:

        logging.error(f"Query failed: {q} | Error: {e}")

        print("Error occurred while processing query.")


# -----------------------------
# Save results
# -----------------------------
output_path = BASE_DIR / "outputs"
output_path.mkdir(exist_ok=True)

file_path = output_path / "rag_answers.json"

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(results_list, f, indent=4)

print("\nResults saved to:", file_path)