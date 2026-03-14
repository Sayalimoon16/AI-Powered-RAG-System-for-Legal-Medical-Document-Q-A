# src/utils.py

import os
import logging
import time
from langdetect import detect


# -----------------------------
# Logging setup
# -----------------------------
def setup_logger():

    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        filename="logs/app.log",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


# -----------------------------
# Environment validation
# -----------------------------
def validate_env():

    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY missing. Add it to .env file."
        )


# -----------------------------
# Empty PDF detection
# -----------------------------
def check_empty_pdf(pages):

    text = ""

    for p in pages:
        text += p["text"]

    if len(text.strip()) == 0:
        raise ValueError(
            "Uploaded PDF has no extractable text. Possibly scanned."
        )


# -----------------------------
# Language detection
# -----------------------------
def detect_language(query):

    try:
        lang = detect(query)

        if lang != "en":
            return False

        return True

    except:
        return True


# -----------------------------
# API Retry with exponential backoff
# -----------------------------
def api_retry(func, retries=3):

    for i in range(retries):

        try:
            return func()

        except Exception as e:

            wait = 2 ** i
            logging.error(f"API Error: {e}")

            time.sleep(wait)

    raise RuntimeError("API failed after retries")


# -----------------------------
# Large document batching
# -----------------------------
def batch_pages(pages, batch_size=50):

    for i in range(0, len(pages), batch_size):
        yield pages[i:i+batch_size]