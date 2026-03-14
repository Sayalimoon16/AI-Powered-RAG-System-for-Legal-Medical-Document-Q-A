import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

from src.prompt_templates import LEGAL_PROMPT, MEDICAL_PROMPT
# -----------------------
# Load .env
# -----------------------
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")


# -----------------------
# Groq Client
# -----------------------
client = Groq(api_key=API_KEY)

MODEL = "llama-3.1-8b-instant"


# -----------------------
# Build Context
# -----------------------
def build_context(chunks):

    context = ""

    for c in chunks:
        context += f"{c['text']}\n(Source: {c['source']}, Page {c['page']})\n\n"

    return context


# -----------------------
# Generate Answer
# -----------------------
def generate_answer(question, retrieved_chunks, domain_mode):

    # No results guard
    if not retrieved_chunks or retrieved_chunks[0] == "Not enough information found":
        return "The document does not contain information about this."

    context = build_context(retrieved_chunks)

    # -----------------------
    # Select prompt based on domain
    # -----------------------
    if domain_mode == "Legal":
        prompt_template = LEGAL_PROMPT

    elif domain_mode == "Medical":
        prompt_template = MEDICAL_PROMPT

    else:
        # fallback
        doc_type = retrieved_chunks[0].get("doc_type", "legal")

        if doc_type == "medical":
            prompt_template = MEDICAL_PROMPT
        else:
            prompt_template = LEGAL_PROMPT


    prompt = prompt_template.format(
        context=context,
        question=question
    )


    # -----------------------
    # Call Groq LLM
    # -----------------------
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer only from the given document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content