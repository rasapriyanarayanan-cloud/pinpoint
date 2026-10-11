"""PinPoint: Multilingual Document Q&A with Verifiable Sources (mobile-friendly UI).

Run on this computer:      streamlit run app.py
Open from a phone (same Wi-Fi):  streamlit run app.py --server.address 0.0.0.0
See DEPLOY_MOBILE.md for putting it online so it works on any phone.
"""
import hashlib

import streamlit as st

from pinpoint_core import (
    Retriever,
    answer_question,
    chunk_pages,
    detect_provider,
    highlight_page,
    load_embedder,
    load_pdf_pages,
)

# Centered layout + collapsed sidebar = one clean column that fits a phone screen.
st.set_page_config(page_title="PinPoint", page_icon="📍", layout="centered", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    .block-container {padding: 1.2rem 1rem calc(3rem + env(safe-area-inset-bottom, 0px)); max-width: 760px;}
    h1 {font-size: 1.9rem !important; margin-bottom: 0;}
    /* big, full-width tap targets */
    form button, .stButton button {width: 100%; min-height: 3rem; font-size: 1.05rem; border-radius: 12px;}
    /* 16px inputs stop iOS Safari from zooming the page on focus */
    input, textarea {font-size: 16px !important;}
    [data-testid="stImage"] img {border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading multilingual embedding model (first run downloads it)...")
def get_embedder():
    return load_embedder()


@st.cache_data(show_spinner=False, max_entries=24)
def render_source(pdf_bytes, page, text):
    """Cache page renders so tapping around does not re-draw the PDF every time."""
    return highlight_page(pdf_bytes, page, text, zoom=1.3)


# ------------------------------------------------------------------ header
st.title("📍 PinPoint")
st.caption("Multilingual Document Q&A with Verifiable Sources")

# ------------------------------------------------------------------ settings (in-page, not a hidden sidebar)
with st.expander("⚙️ Settings"):
    providers = ["none", "claude", "gemini"]
    default = detect_provider()
    provider = st.selectbox(
        "Answer engine",
        providers,
        index=providers.index(default),
        format_func=lambda p: {
            "none": "No LLM (extractive, works offline)",
            "claude": "Claude (grounded generation)",
            "gemini": "Gemini (grounded generation)",
        }[p],
    )
    api_key = None
    if provider != "none":
        api_key = st.text_input("API key (optional if set as environment variable)", type="password") or None
    top_k = st.slider("Passages retrieved (top-k)", 1, 8, 4)
    min_score = st.slider(
        "Relevance threshold",
        0.0,
        0.8,
        0.25,
        0.01,
        help="Questions whose best match scores below this are answered 'Not found in the document'.",
    )
    chunk_size = st.slider("Chunk size (characters)", 200, 1000, 500, 50)
    show_scores = st.checkbox("Show similarity scores", value=True)

# ------------------------------------------------------------------ upload
uploaded = st.file_uploader("Upload a PDF document", type="pdf", help="On a phone, tap 'Browse files' and pick a PDF from your Files or Drive.")

if uploaded is None:
    st.info("Upload a PDF, then ask a question in any language. Every answer shows the exact page it came from.")
    st.stop()

pdf_bytes = uploaded.getvalue()
doc_key = hashlib.md5(pdf_bytes).hexdigest() + f"-{chunk_size}"

if st.session_state.get("doc_key") != doc_key:
    with st.spinner("Reading and indexing the document..."):
        pages = load_pdf_pages(pdf_bytes)
        chunks = chunk_pages(pages, size=chunk_size, overlap=chunk_size // 5)
        if not chunks:
            st.error("No readable text found. This looks like a scanned PDF (image-only); PinPoint needs selectable text.")
            st.stop()
        st.session_state.retriever = Retriever(get_embedder(), chunks)
        st.session_state.doc_key = doc_key
        st.session_state.n_pages = len(pages)
        # a new document invalidates the previous question and answer
        for stale in ("question", "answer", "answer_key"):
            st.session_state.pop(stale, None)

retriever = st.session_state.retriever
st.caption(f"📄 {st.session_state.n_pages} pages  ·  🧩 {len(retriever.chunks)} chunks  ·  🤖 {provider}")

# ------------------------------------------------------------------ Q&A
# A form gives one clear "Ask" button and the phone keyboard's Go key submits it.
with st.form("ask_form"):
    typed = st.text_input("Ask a question about this document (any language)", placeholder="e.g. What is the eligibility criteria?")
    submitted = st.form_submit_button("Ask PinPoint", type="primary")

if submitted and typed.strip():
    st.session_state.question = typed.strip()

question = st.session_state.get("question")
if not question:
    st.stop()

# Only call the retriever / LLM when the question or settings change, not on every tap.
cache_key = (doc_key, question, top_k, min_score, provider)
if st.session_state.get("answer_key") != cache_key:
    with st.spinner("Searching and generating a grounded answer..."):
        try:
            st.session_state.answer = answer_question(
                question, retriever, k=top_k, min_score=min_score, provider=provider, api_key=api_key
            )
        except Exception as exc:  # network / key / quota problems should not crash the app
            st.error(f"The answer engine failed ({exc}). Switch to 'No LLM' in Settings to continue.")
            st.stop()
    st.session_state.answer_key = cache_key

result = st.session_state.answer
st.markdown(f"**Q:** {question}")

if result["grounded"]:
    st.success(result["answer"])
else:
    st.warning(result["answer"])
    st.caption("The document does not contain a confident answer, so PinPoint declined instead of guessing.")

if result["grounded"]:
    st.subheader("Verifiable sources")
    for rank, hit in enumerate(result["hits"], start=1):
        label = f"Source {rank}: page {hit['page']}"
        if show_scores:
            label += f"  (similarity {hit['score']:.2f})"
        with st.expander(label, expanded=(rank == 1)):
            st.markdown(hit["text"])
            png, found = render_source(pdf_bytes, hit["page"], hit["text"])
            st.image(png, caption=f"Page {hit['page']}" + ("" if found else " (passage could not be highlighted)"))
