"""Claude API response layer for conversational delivery."""

import os
import anthropic

SYSTEM_PROMPT = """You are a helpful customer support assistant for Thoughtful AI,
a company that provides AI-powered automation agents for healthcare revenue cycle
management. Their key products include EVA (eligibility verification), CAM (claims
processing), and PHIL (payment posting).

If asked about something unrelated to Thoughtful AI, answer helpfully but briefly,
and gently guide the conversation back to how Thoughtful AI can help with healthcare
automation."""


def _get_client() -> anthropic.Anthropic | None:
    """Get Anthropic client if API key is available."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def generate_response(user_input: str, chat_history: list, matched_answer: str | None = None) -> str:
    """Generate a conversational response using Claude.

    If matched_answer is provided, Claude uses it as source material
    and delivers it conversationally (RAG-style). If not, Claude
    answers freely as a Thoughtful AI support agent.
    """
    client = _get_client()

    # No API key: return matched answer directly or a helpful fallback
    if not client:
        if matched_answer:
            return matched_answer
        return ("I don't have enough information to answer that specific question. "
                "Could you try asking about our agents EVA, CAM, or PHIL? "
                "You can also visit thoughtful.ai for more details.")

    try:
        system = SYSTEM_PROMPT
        if matched_answer:
            system += (f"\n\nUse the following verified information to answer the user's "
                       f"question. Keep your response grounded in this data but deliver "
                       f"it naturally and conversationally. Do not make up additional "
                       f"details beyond what is provided:\n\n{matched_answer}")

        messages = []
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system,
            messages=messages
        )
        return response.content[0].text

    except Exception:
        # If Claude fails but we have a matched answer, still return it
        if matched_answer:
            return matched_answer
        return ("I'm having trouble connecting right now. "
                "Could you try asking about our agents EVA, CAM, or PHIL? "
                "You can also visit thoughtful.ai for more details.")
