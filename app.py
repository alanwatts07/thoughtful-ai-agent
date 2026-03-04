"""Thoughtful AI Support Agent — Streamlit Chat UI."""

import streamlit as st
from matcher import find_best_match
from fallback import generate_response

st.set_page_config(
    page_title="Thoughtful AI Support",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Thoughtful AI Support Agent")
st.caption("Ask me about EVA, CAM, PHIL, or anything else about Thoughtful AI.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "confidence" in message:
            st.caption(f"🎯 Match confidence: {message['confidence']:.1%}")
        elif "source" in message:
            st.caption(f"💬 {message['source']}")

# Handle user input
if prompt := st.chat_input("Ask a question about Thoughtful AI..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Find best match from predefined Q&A
    result = find_best_match(prompt)
    history = [m for m in st.session_state.messages if "confidence" not in m and "source" not in m]

    if result["is_match"]:
        # Match found: pass matched answer through Claude for conversational delivery
        response = generate_response(prompt, history, matched_answer=result["answer"])
        confidence = result["confidence"]
        source = "Verified answer"
    else:
        # No match: let Claude answer freely
        response = generate_response(prompt, history)
        confidence = None
        source = "General response via Claude"

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
        if confidence is not None:
            st.caption(f"🎯 Match confidence: {confidence:.1%}")
        else:
            st.caption(f"💬 {source}")

    msg = {"role": "assistant", "content": response}
    if confidence is not None:
        msg["confidence"] = confidence
    msg["source"] = source
    st.session_state.messages.append(msg)
