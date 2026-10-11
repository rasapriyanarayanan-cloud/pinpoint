"""
PinPoint core: multilingual Retrieval-Augmented Generation with verifiable sources.

Pipeline
    PDF -> page text -> chunks (never cross a page, so every source page is exact)
        -> multilingual sentence embeddings -> FAISS cosine search
        -> grounded LLM answer (or extractive fallback) + page-level source highlight
"""
import os
import re

import numpy as np

# Multilingual model: handles 50+ languages including Tamil, Hindi, Telugu, English.
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

NOT_FOUND = "Not found in the document."

SYSTEM_PROMPT = (
    "You are PinPoint, a careful document assistant. Answer the user's question using ONLY "
    "the numbered context passages provided. Rules:\n"
    "1. Reply in the same language as the question.\n"
    "2. Be concise (2-5 sentences).\n"
    "3. After each fact, cite the page like [Page 3].\n"
    f"4. If the context does not contain the answer, reply with exactly: {NOT_FOUND}\n"
    "5. Never use outside knowledge and never guess."
)

PLAIN_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question concisely in the language of the question. "
    "If you do not know, say so."
)


# --------------------------------------------------------------------------- PDF + chunking
def load_pdf_pages(pdf_bytes):
    """Return a list of (page_number, text) using 1-based page numbers."""
    import pymupdf as fitz  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    doc.close()
    return pages


def chunk_pages(pages, size=500, overlap=100):
    """Split each page into overlapping chunks, preferring sentence boundaries.

    Chunks never span two pages, so the cited page number is always exact.
    """
    chunks = []
    for page_no, raw in pages:
        text = re.sub(r"\s+", " ", raw).strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                cut = max(
                    text.rfind(". ", start, end),
                    text.rfind("? ", start, end),
                    text.rfind("! ", start, end),
                    text.rfind("। ", start, end),  # Devanagari danda
                )
                if cut > start + size // 2:
                    end = cut + 1
            piece = text[start:end].strip()
            if len(piece) > 20:
                chunks.append({"id": len(chunks), "page": page_no, "text": piece})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


# --------------------------------------------------------------------------- retrieval
def load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


class Retriever:
    """Cosine-similarity search over chunk embeddings (FAISS if installed, NumPy otherwise)."""

    def __init__(self, model, chunks):
        self.model = model
        self.chunks = chunks
        self.emb = np.asarray(
            model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )
        try:
            import faiss

            self.index = faiss.IndexFlatIP(self.emb.shape[1])
            self.index.add(self.emb)
        except ImportError:
            self.index = None

    def search(self, query, k=4):
        qv = np.asarray(
            self.model.encode([query], normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )
        k = min(k, len(self.chunks))
        if self.index is not None:
            scores, ids = self.index.search(qv, k)
            pairs = zip(ids[0], scores[0])
        else:
            sims = self.emb @ qv[0]
            top = np.argsort(-sims)[:k]
            pairs = ((i, sims[i]) for i in top)
        return [{**self.chunks[int(i)], "score": float(s)} for i, s in pairs if int(i) >= 0]


# --------------------------------------------------------------------------- LLM calls
def detect_provider():
    """Pick a provider from whichever API key is present in the environment."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "none"


def call_llm(prompt, system, provider, api_key=None):
    if provider == "claude":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
            max_tokens=700,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    if provider == "gemini":
        from google import genai
        from google.genai import types

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.1),
        )
        return (resp.text or "").strip()
    raise ValueError(f"Unknown provider: {provider}")


def answer_question(question, retriever, k=4, min_score=0.25, provider="none", api_key=None):
    """Return {"answer", "hits", "grounded", "mode"}.

    grounded=False means the document did not contain a usable answer.
    """
    hits = retriever.search(question, k=k)
    if not hits or hits[0]["score"] < min_score:
        return {"answer": NOT_FOUND, "hits": hits, "grounded": False, "mode": "threshold"}

    if provider == "none":
        best = hits[0]
        return {
            "answer": f"{best['text']}\n\n(Extractive mode: showing the best-matching passage. Add an API key for a written answer.)",
            "hits": hits,
            "grounded": True,
            "mode": "extractive",
        }

    context = "\n\n".join(f"[Page {h['page']}] {h['text']}" for h in hits)
    prompt = f"Context passages:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    answer = call_llm(prompt, SYSTEM_PROMPT, provider, api_key)
    grounded = NOT_FOUND.lower().rstrip(".") not in answer.lower()
    return {"answer": answer, "hits": hits, "grounded": grounded, "mode": provider}


def plain_llm_answer(question, provider, api_key=None):
    """Baseline for evaluation: the same LLM with NO document access."""
    return call_llm(f"Question: {question}\n\nAnswer:", PLAIN_SYSTEM_PROMPT, provider, api_key)


# --------------------------------------------------------------------------- source highlight
def highlight_page(pdf_bytes, page_no, chunk_text, zoom=1.6):
    """Render a PDF page as PNG with the retrieved passage highlighted (best effort).

    Returns (png_bytes, found) where found says whether any text was located on the page.
    """
    import pymupdf as fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_no - 1]
    found = False
    for sentence in re.split(r"(?<=[.!?।])\s+", chunk_text):
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue
        for needle in (sentence, sentence[:80], sentence[:40]):
            rects = page.search_for(needle)
            if rects:
                for r in rects:
                    page.add_highlight_annot(r)
                found = True
                break
    png = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")
    doc.close()
    return png, found
