"""Semantic similarity matching engine using TF-IDF + cosine similarity."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data import QUESTIONS

CONFIDENCE_THRESHOLD = 0.35

# Keyword map: if any of these appear in user input, force match to that question index
KEYWORD_MAP = {
    "eva": 0,       # Eligibility Verification Agent
    "cam": 1,       # Claims Processing Agent
    "phil": 2,      # Payment Posting Agent
}

# Pre-compute TF-IDF vectors for all predefined questions
question_texts = [q["question"] for q in QUESTIONS]
vectorizer = TfidfVectorizer(stop_words="english")
question_vectors = vectorizer.fit_transform(question_texts)


def _keyword_match(user_input: str) -> dict | None:
    """Check for exact agent name keywords before semantic matching."""
    # Strip punctuation so "PHIL?" and "PHIL." still match
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in user_input)
    words = cleaned.upper().split()
    for keyword, idx in KEYWORD_MAP.items():
        if keyword.upper() in words:
            return {
                "answer": QUESTIONS[idx]["answer"],
                "matched_question": QUESTIONS[idx]["question"],
                "confidence": 1.0,
                "is_match": True
            }
    return None


def find_best_match(user_input: str) -> dict:
    """Find the most relevant predefined answer.

    First checks for exact agent name keywords (EVA, CAM, PHIL).
    Falls back to TF-IDF cosine similarity matching.
    Returns dict with 'answer', 'matched_question', 'confidence',
    and 'is_match' (True if matched).
    """
    # Fast path: keyword match on agent names
    keyword_result = _keyword_match(user_input)
    if keyword_result:
        return keyword_result

    # TF-IDF cosine similarity matching
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, question_vectors)[0]

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    return {
        "answer": QUESTIONS[best_idx]["answer"],
        "matched_question": QUESTIONS[best_idx]["question"],
        "confidence": round(best_score, 3),
        "is_match": best_score >= CONFIDENCE_THRESHOLD
    }
