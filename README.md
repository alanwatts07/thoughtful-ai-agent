# Thoughtful AI Support Agent

A conversational AI support agent that uses **semantic similarity matching** to answer questions about Thoughtful AI's product suite, with **Claude API fallback** for general queries.

## How It Works

1. User asks a question in the chat interface
2. The question is converted to an embedding using `all-MiniLM-L6-v2`
3. Cosine similarity is computed against predefined Q&A about Thoughtful AI's agents (EVA, CAM, PHIL)
4. If confidence is above the threshold, the predefined answer is returned with a confidence score
5. If below threshold, the question is routed to Claude (Haiku) for a contextual response
6. All responses are displayed in a clean Streamlit chat UI with conversation history

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key for Claude fallback (optional — app works without it)
export ANTHROPIC_API_KEY=your_key_here

# Run the app
streamlit run app.py
```

## Project Structure

```
├── app.py           # Streamlit chat UI and conversation logic
├── matcher.py       # Semantic similarity matching engine
├── fallback.py      # Claude API fallback for unmatched queries
├── data.py          # Predefined Q&A dataset
├── requirements.txt # Python dependencies
├── PLAN.md          # Build plan with architecture diagram
└── README.md
```

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| UI | Streamlit | Fastest Python-native chat UI |
| Matching | sentence-transformers | Semantic similarity understands meaning, not just keywords |
| Embeddings | all-MiniLM-L6-v2 | Lightweight, fast, well-suited for small datasets |
| Fallback | Claude API (Haiku) | Fast and cheap for general-purpose responses |

## Design Decisions

- **Semantic matching over keyword matching**: A user asking "How does EVA work?" matches the predefined question about EVA even though the wording is different. This is more robust than string matching or regex.
- **Confidence threshold**: Configurable threshold (default 0.45) determines when to use a predefined answer vs. falling back to Claude. This prevents low-confidence matches from returning incorrect answers.
- **Graceful degradation**: If no API key is set, the fallback returns a helpful message directing users to ask about known topics. The app never crashes on missing config.
- **Conversation history**: Chat history is maintained in Streamlit session state, giving Claude context for follow-up questions in fallback mode.
