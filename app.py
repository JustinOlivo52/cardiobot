import streamlit as st
from config import APP_TITLE, APP_SUBTITLE, DISCLAIMER, check_required_keys, get_missing_required_keys

st.set_page_config(page_title="CardioBot", page_icon="🫀", layout="wide")

with st.sidebar:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.divider()
    key_status = check_required_keys()
    st.subheader("🔑 Model Status")
    st.write("✅ Gemini"  if key_status["gemini"] else "❌ Gemini (add GOOGLE_API_KEY)")
    st.write("✅ Claude"  if key_status["claude"] else "❌ Claude (add ANTHROPIC_API_KEY)")
    st.write("✅ GPT-4"   if key_status["openai"] else "❌ GPT-4 (add OPENAI_API_KEY)")
    st.write("⚪ Ollama (optional)")
    st.divider()
    st.warning(DISCLAIMER)

st.title(APP_TITLE)
st.subheader("Phase 1 Complete ✅")

missing = get_missing_required_keys()
if missing:
    st.error("**Missing API Keys:** Add these to your Codespaces secrets")
    for key in missing:
        st.write(f"  • {key}")
else:
    st.success("✅ All API keys configured! Ready to build Phase 2.")

st.divider()
tab1, tab2, tab3, tab4 = st.tabs(["💬 Clinical Q&A","📊 EKG Interpreter","💊 Drug Calculator","ℹ️ About"])
with tab1: st.info("🔨 Coming in Phase 3")
with tab2: st.info("🔨 Coming in Phase 5")
with tab3: st.info("🔨 Coming in Phase 6")
with tab4:
    st.markdown("""
    ### About CardioBot
    | Model | Role |
    |---|---|
    | 🔵 Claude | Clinical reasoning |
    | 🟢 Gemini | PDF embeddings + EKG vision |
    | 🟠 GPT-4 | Drug dosing |
    | ⚪ Ollama | Local fallback |
    """)
