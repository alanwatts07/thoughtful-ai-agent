# Thoughtful AI Support Agent

A conversational AI support agent that uses **semantic similarity matching** to answer questions about Thoughtful AI's product suite, with **Claude API fallback** for general queries.

## How It Works

1. User asks a question in the chat interface
2. **Keyword check**: if an agent name (EVA, CAM, PHIL) appears in the input, instant match
3. **TF-IDF similarity**: question is vectorized and compared against predefined Q&A using cosine similarity
4. If confidence is above the threshold, the matched answer is passed to Claude as context for **conversational delivery** (RAG-style)
5. If below threshold, Claude answers freely as a Thoughtful AI support agent
6. If no API key is set, matched answers are returned directly and unmatched queries get a helpful fallback
7. All responses are displayed in a Streamlit chat UI with conversation history

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
| Matching | TF-IDF + cosine similarity | Semantic matching without heavy ML dependencies |
| Vectorizer | scikit-learn TfidfVectorizer | Lightweight, no GPU needed, deploys anywhere |
| Response | Claude API (Haiku) | Conversational delivery of matched answers + free-form fallback |

## Design Decisions

- **Three-layer matching**: Keyword match (agent names) → TF-IDF cosine similarity → Claude fallback. Each layer catches what the previous one misses.
- **RAG-style response generation**: Matched predefined answers are passed to Claude as grounding context, not returned as raw strings. This makes every response feel conversational while staying accurate.
- **TF-IDF over heavy embeddings**: For 5 predefined questions, TF-IDF with scikit-learn gives accurate matching without PyTorch (~2GB) or sentence-transformers. Right tool for the scale.
- **Confidence threshold**: Configurable threshold (default 0.35) determines when to use a predefined answer vs. falling back to Claude. Prevents low-confidence matches from returning incorrect answers.
- **Graceful degradation**: If no API key is set, matched answers are returned directly. Unmatched queries get a helpful message. The app never crashes on missing config.
- **Conversation history**: Chat history is maintained in Streamlit session state, giving Claude context for follow-up questions.
