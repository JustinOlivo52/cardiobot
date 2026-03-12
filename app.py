import streamlit as st
from config import APP_TITLE, APP_SUBTITLE, DISCLAIMER, check_required_keys
from agents.claude_agent import ask_cardiobot
from agents.ekg_agent import interpret_ekg
from agents.dosing_agent import get_dosing_guidance
from agents.doc_checker import check_citations
from tools.calculator import get_available_drugs
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

tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Clinical Q&A",
    "📊 EKG Interpreter",
    "💊 Drug Calculator",
    "📚 Citation Checker"
])

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
                    st.warning("⚠️ This interpretation is for educational purposes only.")
                except Exception as e:
                    st.error(f"EKG interpretation failed: {e}")
    else:
        st.info("👆 Upload an EKG image to get started.")

# ── TAB 3: Drug Calculator ───────────────────────────────────────
with tab3:
    st.subheader("💊 Cardiac Drug Dosing Calculator")
    st.caption("Weight-based dosing guidance powered by GPT-4")

    col1, col2 = st.columns(2)
    with col1:
        drug = st.selectbox(
            "Select Drug",
            options=get_available_drugs(),
            help="Choose a cardiac medication"
        )
    with col2:
        weight = st.number_input(
            "Patient Weight (kg)",
            min_value=1.0,
            max_value=300.0,
            value=70.0,
            step=0.5
        )

    context = st.text_input(
        "Clinical Context (optional)",
        placeholder="e.g. STEMI patient in cardiogenic shock, age 65"
    )

    if st.button("💊 Calculate Dose", type="primary"):
        with st.spinner("⚕️ GPT-4 calculating dosing guidance..."):
            try:
                result = get_dosing_guidance(drug, weight, context)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"✅ Dosing guidance for {result['drug']}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Drug", result["drug"])
                    with col2:
                        st.metric("Patient Weight", f"{result['weight_kg']} kg")
                    with col3:
                        st.metric("Calculated Bolus", result["calculated_bolus"])
                    st.divider()
                    st.markdown("### 📋 Clinical Guidance")
                    st.markdown(result["guidance"])
                    st.warning("⚠️ Always verify doses with pharmacy and institutional protocols.")
            except Exception as e:
                st.error(f"Error: {e}")

# ── TAB 4: Citation Checker ──────────────────────────────────────
with tab4:
    st.subheader("📚 Citation Checker")
    st.caption("Search the ESC guidelines directly and see exactly where information comes from")

    search_query = st.text_input(
        "Search Guidelines",
        placeholder="e.g. troponin rule-out algorithm, STEMI door-to-balloon time"
    )

    n_results = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if st.button("🔍 Search Guidelines", type="primary"):
        if not search_query:
            st.warning("Please enter a search query")
        else:
            with st.spinner("🔍 Searching guidelines..."):
                try:
                    citations = check_citations(search_query, n_results=n_results)

                    if not citations:
                        st.warning("No results found")
                    else:
                        st.success(f"Found {len(citations)} relevant passages")
                        st.divider()

                        for cite in citations:
                            with st.expander(
                                f"#{cite['rank']} — {cite['confidence']} confidence "
                                f"(score: {cite['score']}) — Page {cite['page']}"
                            ):
                                st.caption(f"**Source:** {cite['source']}")
                                st.caption(f"**Section:** {cite['heading']}")
                                st.caption(f"**Relevance Score:** {cite['score']}")
                                st.divider()
                                st.markdown(cite['excerpt'])

                except Exception as e:
                    st.error(f"Citation search failed: {e}")
