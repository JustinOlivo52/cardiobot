import streamlit as st
from config import APP_TITLE, APP_SUBTITLE, DISCLAIMER, check_required_keys
from agents.consult_agent import run_consult
from agents.ekg_agent import interpret_ekg
from agents.dosing_agent import get_dosing_guidance
from agents.doc_checker import check_citations
from tools.calculator import get_available_drugs
from memory.conversation import ConversationMemory

st.set_page_config(
    page_title="CardioBot",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_consult" not in st.session_state:
    st.session_state.last_consult = None

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🫀 CardioBot")
    st.markdown("*AI-Powered Clinical Cardiology Assistant*")
    st.divider()

    key_status = check_required_keys()
    all_systems_go = all(v for k, v in key_status.items() if k != "ollama")
    st.markdown("🟢 System Online" if all_systems_go else "🔴 System Offline")

    st.divider()

    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("In Memory", len(st.session_state.memory))

    st.divider()
    st.error("⚠️ **DISCLAIMER**\n\n" + DISCLAIMER)

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.messages = []
        st.session_state.last_consult = None
        st.rerun()

# ── Main Header ──────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align: center; color: #E63946;'>🫀 CardioBot</h1>
    <p style='text-align: center; color: #8B949E; font-size: 1.1em;'>
        AI-Powered Clinical Cardiology Assistant
    </p>
    <p style='text-align: center; color: #8B949E; font-size: 0.85em;'>
        Claude • Gemini • GPT-4 • ESC Guidelines • LangGraph
    </p>
""", unsafe_allow_html=True)
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Cardiology Consult",
    "💬 Clinical Q&A",
    "📊 EKG Interpreter",
    "💊 Drug Calculator",
    "📚 Citation Checker"
])

# ── TAB 1: Cardiology Consult ────────────────────────────────────
with tab1:
    st.markdown("### 🩺 Cardiology Consult")
    st.caption("Present your patient naturally — CardioBot will generate a full consult report grounded in ESC guidelines")

    st.markdown("""
    **Example presentations:**
    - *"65yo male with 2 hours of crushing chest pain radiating to left arm, diaphoretic, BP 90/60, HR 110, EKG shows ST elevation in II, III, aVF"*
    - *"52yo female with atypical chest discomfort, known DM and HTN, troponin mildly elevated at 0.08"*
    """)

    presentation = st.text_area(
        "Patient Presentation",
        placeholder="Describe the patient as you would in a clinical handoff...",
        height=120
    )

    if st.button("🩺 Run Cardiology Consult", type="primary", use_container_width=True):
        if not presentation.strip():
            st.warning("Please enter a patient presentation")
        else:
            with st.spinner("🫀 CardioBot consulting ESC guidelines..."):
                try:
                    result = run_consult(presentation)
                    st.session_state.last_consult = result
                    st.session_state.messages.append({
                        "role": "user",
                        "content": presentation
                    })
                    st.session_state.memory.add_user_message(presentation)
                    st.session_state.memory.add_assistant_message(result["report"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["report"],
                        "sources": result["sources"],
                        "scores": result["scores"]
                    })
                except Exception as e:
                    st.error(f"Consult failed: {str(e)[:200]}")

    # Display consult report
    if st.session_state.last_consult:
        st.divider()
        st.markdown(st.session_state.last_consult["report"])
        st.divider()
        with st.expander("📚 Guideline Sources"):
            for src, score in zip(
                st.session_state.last_consult["sources"],
                st.session_state.last_consult["scores"]
            ):
                st.caption(f"• {src} (relevance: {score})")
        st.warning("⚠️ For educational purposes only. Not a substitute for clinical judgment.")
        
# ── TAB 2: Clinical Q&A ──────────────────────────────────────────
with tab2:
    st.markdown("### 💬 Clinical Q&A")
    st.caption("Ask follow-up questions about cardiology — powered by Claude + ESC Guidelines")

    question = st.chat_input("Ask a cardiology question...")

    chat_container = st.container()
    with chat_container:
        for msg in reversed(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                    with st.expander("📚 Sources"):
                        for src, score in zip(msg["sources"], msg["scores"]):
                            st.caption(f"• {src} (relevance: {score})")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("🫀 Consulting guidelines..."):
            try:
                from agents.claude_agent import ask_cardiobot
                history = st.session_state.memory.get_history()
                result = ask_cardiobot(question, conversation_history=history)
                st.session_state.memory.add_user_message(question)
                st.session_state.memory.add_assistant_message(result["answer"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "scores": result["scores"]
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {str(e)[:200]}",
                    "sources": [],
                    "scores": []
                })
        st.rerun()

# ── TAB 3: EKG Interpreter ───────────────────────────────────────
with tab3:
    st.markdown("### 📊 EKG / ECG Interpreter")
    st.caption("Upload an EKG strip for AI-powered interpretation using Gemini Vision")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload EKG image",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )
    with col2:
        st.markdown("**Supported formats:**")
        st.markdown("• JPG / JPEG\n• PNG\n• BMP\n• WebP")
        st.markdown("**Max size:** 10MB")

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded EKG", use_container_width=True)
        st.divider()
        if st.button("🔍 Interpret EKG", type="primary", use_container_width=True):
            with st.spinner("🧠 Gemini Vision analyzing EKG..."):
                try:
                    result = interpret_ekg(uploaded_file)
                    st.success("✅ Interpretation complete")
                    st.markdown(result["interpretation"])
                    st.divider()
                    st.caption(f"Analyzed by: {result['model']}")
                    st.warning("⚠️ For educational purposes only.")
                except Exception as e:
                    st.error(f"EKG interpretation failed: {str(e)[:200]}")
    else:
        st.info("👆 Upload an EKG image to get started.")

# ── TAB 4: Drug Calculator ───────────────────────────────────────
with tab4:
    st.markdown("### 💊 Cardiac Drug Dosing Calculator")
    st.caption("Weight-based dosing guidance powered by GPT-4")

    col1, col2 = st.columns(2)
    with col1:
        drug = st.selectbox("Select Drug", options=get_available_drugs())
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

    if st.button("💊 Calculate Dose", type="primary", use_container_width=True):
        with st.spinner("⚕️ GPT-4 calculating dosing guidance..."):
            try:
                result = get_dosing_guidance(drug, weight, context)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"✅ Dosing guidance for **{result['drug']}**")
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
                    st.warning("⚠️ Always verify with pharmacy and institutional protocols.")
            except Exception as e:
                st.error(f"Dosing calculation failed: {str(e)[:200]}")

# ── TAB 5: Citation Checker ──────────────────────────────────────
with tab5:
    st.markdown("### 📚 Citation Checker")
    st.caption("Search 895 indexed passages from the 2023 ESC ACS Guidelines")

    search_query = st.text_input(
        "Search Guidelines",
        placeholder="e.g. troponin rule-out algorithm, door-to-balloon time"
    )

    col1, col2 = st.columns([3, 1])
    with col2:
        n_results = st.slider("Results", min_value=1, max_value=10, value=5)

    if st.button("🔍 Search Guidelines", type="primary", use_container_width=True):
        if not search_query:
            st.warning("Please enter a search query")
        else:
            with st.spinner("🔍 Searching ESC guidelines..."):
                try:
                    citations = check_citations(search_query, n_results=n_results)
                    if not citations:
                        st.warning("No results found.")
                    else:
                        st.success(f"Found {len(citations)} relevant passages")
                        st.divider()
                        for cite in citations:
                            with st.expander(
                                f"#{cite['rank']} — {cite['confidence']} confidence "
                                f"| Score: {cite['score']} | Page {cite['page']}"
                            ):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.caption(f"**Source:** {cite['source']}")
                                with col2:
                                    st.caption(f"**Section:** {cite['heading']}")
                                st.divider()
                                st.markdown(cite['excerpt'])
                except Exception as e:
                    st.error(f"Citation search failed: {str(e)[:200]}")
