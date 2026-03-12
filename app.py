import streamlit as st
from config import APP_TITLE, APP_SUBTITLE, DISCLAIMER, check_required_keys
from agents.claude_agent import ask_cardiobot
from agents.ekg_agent import interpret_ekg
from memory.conversation import ConversationMemory

st.set_page_config(page_title="CardioBot", page_icon="🫀", layout="wide")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "messages" not in st.session_state:
    st.session_state.messages = []

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
    st.metric("Messages in memory", len(st.session_state.memory))
    st.divider()
    st.warning(DISCLAIMER)
    if st.button("🗑️ Clear Chat"):
        st.session_state.memory.clear()
        st.session_state.messages = []
        st.rerun()

st.title("🫀 CardioBot")
st.caption("AI-Powered Clinical Cardiology Assistant — Powered by Claude + ESC Guidelines")
st.divider()

tab1, tab2 = st.tabs(["💬 Clinical Q&A", "📊 EKG Interpreter"])

# ── TAB 1: Clinical Q&A ──────────────────────────────────────────
with tab1:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Sources"):
                    for src, score in zip(msg["sources"], msg["scores"]):
                        st.caption(f"• {src} (relevance: {score})")

    if question := st.chat_input("Ask a cardiology question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🫀 Consulting guidelines..."):
                try:
                    history = st.session_state.memory.get_history()
                    result = ask_cardiobot(question, conversation_history=history)
                    st.session_state.memory.add_user_message(question)
                    st.session_state.memory.add_assistant_message(result["answer"])
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

# ── TAB 2: EKG Interpreter ───────────────────────────────────────
with tab2:
    st.subheader("📊 EKG / ECG Interpreter")
    st.caption("Upload an EKG strip for AI-powered interpretation using Gemini Vision")

    uploaded_file = st.file_uploader(
        "Upload EKG image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Upload a clear photo or scan of an EKG strip"
    )

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded EKG", use_container_width=True)

        if st.button("🔍 Interpret EKG", type="primary"):
            with st.spinner("🧠 Gemini Vision analyzing EKG..."):
                try:
                    result = interpret_ekg(uploaded_file)
                    st.success("✅ Interpretation complete")
                    st.markdown("### 📋 EKG Interpretation")
                    st.markdown(result["interpretation"])
                    st.caption(f"Analyzed by: {result['model']}")
                    st.warning("⚠️ This interpretation is for educational purposes only. Always confirm with a supervising physician.")
                except Exception as e:
                    st.error(f"EKG interpretation failed: {e}")
    else:
        st.info("👆 Upload an EKG image to get started. You can find sample EKG images by searching 'sample ECG strip PNG' online.")
