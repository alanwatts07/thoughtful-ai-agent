"""Thoughtful AI Support Agent — Streamlit Chat UI."""

import streamlit as st
from matcher import find_best_match
from fallback import get_fallback_response

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
            st.caption(f"🎯 Confidence: {message['confidence']:.1%}")

# Handle user input
if prompt := st.chat_input("Ask a question about Thoughtful AI..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Find best match
    result = find_best_match(prompt)

    if result["is_match"]:
        response = result["answer"]
        confidence = result["confidence"]
    else:
        # Fallback to Claude
        history = [m for m in st.session_state.messages if "confidence" not in m]
        response = get_fallback_response(prompt, history)
        confidence = None

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
        if confidence is not None:
            st.caption(f"🎯 Confidence: {confidence:.1%}")
        else:
            st.caption("💬 General response via Claude")

    msg = {"role": "assistant", "content": response}
    if confidence is not None:
        msg["confidence"] = confidence
    st.session_state.messages.append(msg)
