"""Semantic similarity matching engine using sentence-transformers."""

from sentence_transformers import SentenceTransformer, util
from data import QUESTIONS

CONFIDENCE_THRESHOLD = 0.45

model = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-compute embeddings for all predefined questions
question_texts = [q["question"] for q in QUESTIONS]
question_embeddings = model.encode(question_texts, convert_to_tensor=True)


def find_best_match(user_input: str) -> dict:
    """Find the most semantically similar predefined question.

    Returns dict with 'answer', 'matched_question', 'confidence',
    and 'is_match' (True if confidence >= threshold).
    """
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
