# src/ingestion.py

import fitz
from pathlib import Path
from typing import List, Dict
from src.utils import batch_pages, check_empty_pdf


# -----------------------------------
# Domain Keywords
# -----------------------------------
LEGAL_KEYWORDS = [
    "act", "section", "clause", "agreement",
    "contract", "liability", "party"
]

MEDICAL_KEYWORDS = [
    "patient", "diagnosis", "treatment",
    "lab", "hospital", "clinical", "test"
]


# -----------------------------------
# Detect document domain
# -----------------------------------
def detect_doc_type(text: str) -> str:
    """
    Classify document type based on keywords
    """

    text = text.lower()

    legal_score = sum(k in text for k in LEGAL_KEYWORDS)
    medical_score = sum(k in text for k in MEDICAL_KEYWORDS)

    if legal_score > medical_score:
        return "legal"

    elif medical_score > legal_score:
        return "medical"

    return "unknown"


# -----------------------------------
# Extract PDF Text
# -----------------------------------
def extract_pdf_text(pdf_path, original_filename=None) -> List[Dict]:
    """
    Extract text from PDF page by page.
    Supports multi-column layouts and large PDFs.
    """

    pdf_path = Path(pdf_path)

    pages = []

    try:
        doc = fitz.open(pdf_path)

    except Exception as e:
        print(f"[ERROR] Cannot open PDF: {e}")
        return []

    full_text = ""

    # -----------------------------------
    # Batch processing for large PDFs
    # -----------------------------------
    page_indices = list(range(len(doc)))

    for batch in batch_pages(page_indices, batch_size=50):

        for i in batch:

            page = doc.load_page(i)

            # Better extraction for multi-column documents
            blocks = page.get_text("blocks")

            text = " ".join(block[4] for block in blocks)

            text = text.strip()

            full_text += text

            pages.append({
                "page_num": i + 1,
                "source_file": original_filename if original_filename else pdf_path.name,
                "text": text
            })

    # -----------------------------------
    # Empty PDF detection
    # -----------------------------------
    try:
        check_empty_pdf(pages)
    except ValueError as e:
        print(f"[WARNING] {e}")
        return []

    # -----------------------------------
    # Detect document domain
    # -----------------------------------
    doc_type = detect_doc_type(full_text)

    for p in pages:
        p["doc_type"] = doc_type

    return pages


# -----------------------------------
# Batch ingestion (multiple PDFs)
# -----------------------------------
def ingest_folder(folder_path: str) -> List[Dict]:
    """
    Load all PDFs from folder
    """

    folder = Path(folder_path)

    all_pages = []

    for pdf in folder.glob("*.pdf"):

        print(f"[INFO] Processing {pdf.name}")

        pages = extract_pdf_text(pdf)

        all_pages.extend(pages)

    return all_pages


# -----------------------------------
# Test run
# -----------------------------------
if __name__ == "__main__":

    test_file = "data/sample_medical/MedicalReport1.pdf"

    pages = extract_pdf_text(test_file, "MedicalReport1.pdf")

    print("Total pages:", len(pages))

    if pages:
        print("\nFirst page sample:\n")
        print(pages[0])