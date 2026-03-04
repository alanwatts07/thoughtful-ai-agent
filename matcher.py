"""Semantic similarity matching engine using sentence-transformers."""

from sentence_transformers import SentenceTransformer, util
from data import QUESTIONS

CONFIDENCE_THRESHOLD = 0.45

# Keyword map: if any of these appear in user input, force match to that question index
KEYWORD_MAP = {
    "eva": 0,       # Eligibility Verification Agent
    "cam": 1,       # Claims Processing Agent
    "phil": 2,      # Payment Posting Agent
}

model = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-compute embeddings for all predefined questions
question_texts = [q["question"] for q in QUESTIONS]
question_embeddings = model.encode(question_texts, convert_to_tensor=True)


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
    Falls back to semantic similarity matching via embeddings.
    Returns dict with 'answer', 'matched_question', 'confidence',
    and 'is_match' (True if matched).
    """
    # Fast path: keyword match on agent names
    keyword_result = _keyword_match(user_input)
    if keyword_result:
        return keyword_result

    # Semantic similarity matching
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, question_embeddings)[0]

    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()

    return {
        "answer": QUESTIONS[best_idx]["answer"],
        "matched_question": QUESTIONS[best_idx]["question"],
        "confidence": round(best_score, 3),
        "is_match": best_score >= CONFIDENCE_THRESHOLD
    }
