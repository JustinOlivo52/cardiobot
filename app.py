import streamlit as st
from config import APP_TITLE, APP_SUBTITLE, DISCLAIMER, check_required_keys
from agents.graph import run_cardiobot
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
    st.caption("🕸️ Powered by LangGraph — automatically routes to the right tool")

    # Chat input FIRST — this pins it to the bottom
    question = st.chat_input("Ask anything about cardiology...")

    # Chat history container
    chat_container = st.container()
    with chat_container:
        for msg in reversed(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                    with st.expander("📚 Sources"):
                        for src, score in zip(msg["sources"], msg["scores"]):
                            st.caption(f"• {src} (relevance: {score})")

    # Process new question
    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("🕸️ CardioBot thinking..."):
            try:
                history = st.session_state.memory.get_history()
                result = run_cardiobot(question, conversation_history=history)

                if result["type"] == "rag":
                    answer = result["answer"]
                    st.session_state.memory.add_user_message(question)
                    st.session_state.memory.add_assistant_message(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": result["sources"],
                        "scores": result["scores"]
                    })

                elif result["type"] == "citation":
                    citations = result["citations"]
                    answer = f"Found {len(citations)} relevant passages:\n\n"
                    for c in citations[:3]:
                        answer += f"**{c['confidence']}** — Page {c['page']}\n{c['excerpt'][:200]}...\n\n"
                    st.session_state.memory.add_user_message(question)
                    st.session_state.memory.add_assistant_message(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [],
                        "scores": []
                    })

                elif result["type"] in ["dosing_redirect", "ekg_redirect"]:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["message"],
                        "sources": [],
                        "scores": []
                    })

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {e}",
                    "sources": [],
                    "scores": []
                })

        st.rerun()

# ── TAB 2: EKG Interpreter ───────────────────────────────────────
with tab2:
    st.subheader("📊 EKG / ECG Interpreter")
    st.caption("Upload an EKG strip for AI-powered interpretation using Gemini Vision")

    uploaded_file = st.file_uploader(
        "Upload EKG image",
        type=["jpg", "jpeg", "png", "bmp", "webp"]
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
                    st.warning("⚠️ For educational purposes only.")
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
        drug = st.selectbox("Select Drug", options=get_available_drugs())
    with col2:
        weight = st.number_input("Patient Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.5)

    context = st.text_input("Clinical Context (optional)", placeholder="e.g. STEMI patient, age 65")

    if st.button("💊 Calculate Dose", type="primary"):
        with st.spinner("⚕️ GPT-4 calculating..."):
            try:
                result = get_dosing_guidance(drug, weight, context)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"✅ Dosing guidance for {result['drug']}")
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Drug", result["drug"])
                    with col2: st.metric("Weight", f"{result['weight_kg']} kg")
                    with col3: st.metric("Calculated Bolus", result["calculated_bolus"])
                    st.divider()
                    st.markdown("### 📋 Clinical Guidance")
                    st.markdown(result["guidance"])
                    st.warning("⚠️ Always verify with pharmacy and institutional protocols.")
            except Exception as e:
                st.error(f"Error: {e}")

# ── TAB 4: Citation Checker ──────────────────────────────────────
with tab4:
    st.subheader("📚 Citation Checker")
    st.caption("Search the ESC guidelines directly")

    search_query = st.text_input("Search Guidelines", placeholder="e.g. troponin rule-out algorithm")
    n_results = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if st.button("🔍 Search Guidelines", type="primary"):
        if not search_query:
            st.warning("Please enter a search query")
        else:
            with st.spinner("🔍 Searching..."):
                try:
                    citations = check_citations(search_query, n_results=n_results)
                    if not citations:
                        st.warning("No results found")
                    else:
                        st.success(f"Found {len(citations)} relevant passages")
                        for cite in citations:
                            with st.expander(f"#{cite['rank']} — {cite['confidence']} (score: {cite['score']}) — Page {cite['page']}"):
                                st.caption(f"**Source:** {cite['source']}")
                                st.caption(f"**Section:** {cite['heading']}")
                                st.divider()
                                st.markdown(cite['excerpt'])
                except Exception as e:
                    st.error(f"Citation search failed: {e}")
