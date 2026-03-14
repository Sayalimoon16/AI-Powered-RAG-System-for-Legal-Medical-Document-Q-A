# src/evaluator.py

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from datasets import Dataset
import matplotlib.pyplot as plt

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from retriever import retrieve, load_model, load_index
from LLM_chain import generate_answer


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")


# -----------------------------
# LLM Judge (Groq)
# -----------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# -----------------------------
# Embedding model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Load test questions
# -----------------------------
def load_questions():

    path = Path("data/test_questions.json")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Diversity filter (MMR-like)
# -----------------------------
def filter_diverse_chunks(chunks, max_chunks=5):

    unique_chunks = []
    seen_texts = set()

    for c in chunks:

        if isinstance(c, dict) and "text" in c:

            text = c["text"].strip()

            if text not in seen_texts:

                seen_texts.add(text)
                unique_chunks.append(c)

        if len(unique_chunks) >= max_chunks:
            break

    return unique_chunks


# -----------------------------
# RAGAS Evaluation
# -----------------------------
def run_ragas():

    questions = load_questions()

    model = load_model()

    index, metadata, chunk_ids = load_index(
        "output/faiss_indexes",
        "combined_index"
    )

    data = []

    for q in questions:

        query = q["question"]

        # Retrieve chunks
        retrieved = retrieve(
            query,
            index,
            metadata,
            chunk_ids,
            model
        )

        # Apply diversity filter
        retrieved = filter_diverse_chunks(retrieved, max_chunks=2)

        # Generate answer
        answer = generate_answer(query, retrieved)

        if not answer or answer.strip() == "":
            answer = "No answer generated from the document."

        # Extract contexts
        contexts = [
            r["text"] for r in retrieved
            if isinstance(r, dict)
        ]

        if "does not contain information" in answer.lower():
                continue
    

        data.append({
            "question": query,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": q["ground_truth"]
        })

    # Convert to dataset
    dataset = Dataset.from_list(data)

    # -----------------------------
    # Run RAGAS evaluation
    # -----------------------------
    results = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision
        ],
        llm=llm,
        embeddings=embeddings
    )

    df = results.to_pandas()

    scores = df.select_dtypes(include=["float", "int"]).mean().to_dict()

    # -----------------------------
    # Save results
    # -----------------------------
    save_dir = Path("output/eval_results")
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "ragas_scores.json", "w") as f:
        json.dump(scores, f, indent=2)

    print("\nRAGAS Evaluation Scores:")
    print(scores)

    # -----------------------------
    # Plot metrics
    # -----------------------------
    metrics = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(6,4))
    plt.bar(metrics, values)
    plt.title("RAGAS Evaluation Metrics")
    plt.ylim(0,1)
    plt.ylabel("Score")

    chart_path = save_dir / "ragas_scores_chart.png"

    plt.savefig(chart_path)

    print(f"\nChart saved at: {chart_path}")


# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":

    run_ragas()