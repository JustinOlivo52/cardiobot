import streamlit as st
from config import APP_TITLE, APP_SUBTITLE, DISCLAIMER, check_required_keys
from agents.claude_agent import ask_cardiobot

st.set_page_config(page_title="CardioBot", page_icon="🫀", layout="wide")

with st.sidebar:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.divider()
    key_status = check_required_keys()
    st.subheader("🔑 Model Status")
    st.write("✅ Gemini"  if key_status["gemini"] else "❌ Gemini")
    st.write("✅ Claude"  if key_status["claude"] else "❌ Claude")
    st.write("✅ GPT-4"   if key_status["openai"] else "❌ GPT-4")
    st.divider()
    st.warning(DISCLAIMER)
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🫀 CardioBot")
st.caption("AI-Powered Clinical Cardiology Assistant — Powered by Claude + ESC Guidelines")
st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Sources"):
                for src, score in zip(msg["sources"], msg["scores"]):
                    st.caption(f"• {src} (relevance: {score})")

# Chat input
if question := st.chat_input("Ask a cardiology question..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get CardioBot response
    with st.chat_message("assistant"):
        with st.spinner("🫀 Consulting guidelines..."):
            try:
                result = ask_cardiobot(question)
                st.markdown(result["answer"])
                with st.expander("📚 Sources"):
                    for src, score in zip(result["sources"], result["scores"]):
                        st.caption(f"• {src} (relevance: {score})")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "scores": result["scores"]
                })
            except Exception as e:
                st.error(f"Error: {e}")
