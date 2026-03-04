# Thoughtful AI Support Agent — Build Plan

## Overview
A conversational AI support agent for Thoughtful AI that uses semantic similarity matching to answer predefined questions about their product suite, with Claude API fallback for general queries. Built with Python, Streamlit, and sentence-transformers.

## Architecture
```
User Input → Embedding → Cosine Similarity vs Predefined Q&A
                              ↓
                    Score >= threshold?
                     ↓              ↓
                   YES              NO
                     ↓              ↓
              Return match    Claude API fallback
                     ↓              ↓
                  Streamlit Chat UI ←
```

## Build Steps

- [x] Step 1: Project structure and predefined dataset
- [x] Step 2: Semantic similarity matching engine with embeddings
- [x] Step 3: Claude API fallback for unmatched queries
- [x] Step 4: Streamlit chat UI with conversation history
- [x] Step 5: Error handling, confidence display, and README

## Tech Choices

| Component | Choice | Why |
|-----------|--------|-----|
| UI | Streamlit | Fastest Python-native chat UI. No frontend code needed. |
| Matching | sentence-transformers | Semantic similarity, not keyword matching. Understands meaning. |
| Embeddings | all-MiniLM-L6-v2 | Lightweight, fast, good enough for 5 predefined questions. |
| Fallback | Claude API (Haiku) | Fast, cheap, handles anything outside the predefined set. |
| Language | Python | Best ecosystem for AI/ML tooling. |
