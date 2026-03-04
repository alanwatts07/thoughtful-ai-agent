"""Claude API fallback for questions outside the predefined dataset."""

import os
import anthropic

SYSTEM_PROMPT = """You are a helpful customer support assistant for Thoughtful AI,
a company that provides AI-powered automation agents for healthcare revenue cycle
management. Their key products include EVA (eligibility verification), CAM (claims
processing), and PHIL (payment posting).

If asked about something unrelated to Thoughtful AI, answer helpfully but briefly,
and gently guide the conversation back to how Thoughtful AI can help with healthcare
automation."""


def get_fallback_response(user_input: str, chat_history: list) -> str:
    """Get a response from Claude for questions not in the predefined set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ("I don't have enough information to answer that specific question. "
                "Could you try asking about our agents EVA, CAM, or PHIL? "
                "You can also visit thoughtful.ai for more details.")

    try:
        client = anthropic.Anthropic(api_key=api_key)

        messages = []
        for msg in chat_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=messages
        )
        return response.content[0].text

    except Exception as e:
        return (f"I'm having trouble connecting right now. "
                f"Could you try asking about our agents EVA, CAM, or PHIL? "
                f"You can also visit thoughtful.ai for more details.")
