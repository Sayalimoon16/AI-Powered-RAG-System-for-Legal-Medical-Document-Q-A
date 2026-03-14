import sys
import os
import json

# add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retriever import load_model, load_index, retrieve

# load embedding model
model = load_model()

# load FAISS index
index, metadata, chunk_ids = load_index(
    "../output/faiss_indexes",
    "combined_index"
)

queries = [
    "What is contract liability?",
    "Define agreement",
    "Patient diagnosis",
    "Glucose level",
    "Treatment provided",
    "Partnership definition",
    "Blood test result",
    "Legal clause",
    "Discharge condition",
    "Medical history"
]

all_results = {}

for q in queries:
    print("\nQUERY:", q)

    results = retrieve(q, index, metadata, chunk_ids, model, k=5)

    # print first result
    print(results[:1])

    # convert numpy float to normal float
    clean_results = []
    for r in results:
        r["score"] = float(r["score"])
        clean_results.append(r)

    all_results[q] = clean_results


# create output directory if not exists
os.makedirs("../output/eval_results", exist_ok=True)

# save JSON output
with open("../output/eval_results/retrieval_sample_output.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n✅ Retrieval results saved to output/eval_results/retrieval_sample_output.json")