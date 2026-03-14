# src/chunker.py

import json
import uuid
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50


LEGAL_SEPARATORS = [
    "\nSection", "\nClause", "\nArticle", "\nChapter",
    "\n\n", "\n", ". "
]

MEDICAL_SEPARATORS = [
    "\nDiagnosis", "\nClinical", "\nTreatment",
    "\nMedication", "\nLab", "\nTest",
    "\n\n", "\n", ". "
]


def get_separators(doc_type: str):
    if doc_type == "legal":
        return LEGAL_SEPARATORS
    elif doc_type == "medical":
        return MEDICAL_SEPARATORS
    return ["\n\n", "\n", ". "]


def chunk_documents(
    pages_data: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP
) -> List[Dict]:

    all_chunks = []

    for page in pages_data:
        text = page["text"]
        doc_type = page["doc_type"]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=get_separators(doc_type),
            length_function=len
        )

        splits = splitter.split_text(text)
        char_pointer = 0

        for chunk in splits:
            start = text.find(chunk, char_pointer)
            end = start + len(chunk)
            char_pointer = end

            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk,
                "source": page["source_file"],
                "page": page["page_num"],
                "doc_type": doc_type,
                "char_start": start,
                "char_end": end
            })

    return all_chunks


def save_chunks_json(chunks: List[Dict], out_path: str):
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {len(chunks)} chunks → {out_file}")


if __name__ == "__main__":
    from ingestion import ingest_folder

    legal_pages = ingest_folder("../data/sample_legal")
    medical_pages = ingest_folder("../data/sample_medical")

    all_pages = legal_pages + medical_pages

    chunks = chunk_documents(all_pages, chunk_size=500, overlap=50)
    save_chunks_json(chunks, "../output/chunks_500.json")

    print("[DONE] Chunking complete")