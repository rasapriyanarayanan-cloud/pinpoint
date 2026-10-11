# PinPoint: Multilingual Document Q&A with Verifiable Sources

Upload a PDF, ask a question in any language, get an answer grounded in the document, with the exact source page highlighted. If the document does not contain the answer, PinPoint says so instead of guessing.

## How it works
1. **Read and chunk:** PyMuPDF extracts page text; chunks (about 500 characters, 20% overlap) never cross a page, so the cited page is always exact.
2. **Embed:** a multilingual sentence-transformer (`paraphrase-multilingual-MiniLM-L12-v2`) maps questions and chunks into one shared vector space, so a Tamil question can match English text and the reverse.
3. **Retrieve:** FAISS cosine search returns the top-k passages.
4. **Gate:** if the best similarity is below the threshold, the answer is "Not found in the document."
5. **Generate:** the LLM answers using only the retrieved passages, in the question's language, citing pages.
6. **Verify:** the app shows each source passage and the PDF page with the text highlighted.

## Setup (once, about 10 minutes)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```
The first run downloads the embedding model (about 470 MB), so do this on good internet **before Tuesday**.

## API key (optional but recommended)
Without a key the app still works in **extractive mode** (it shows the best-matching passage). For written answers, get a free key and set it:

| Provider | Variable | Windows PowerShell |
|---|---|---|
| Gemini | `GEMINI_API_KEY` | `$env:GEMINI_API_KEY="your-key"` |
| Claude | `ANTHROPIC_API_KEY` | `$env:ANTHROPIC_API_KEY="your-key"` |

You can also paste the key into the sidebar. Model names change over time; override with `GEMINI_MODEL` or `ANTHROPIC_MODEL` if a name is rejected.

## Run
```bash
streamlit run app.py
```

## Measure results for your results slide
1. Copy `questions_template.csv` to `questions.csv` and write 20 to 30 questions about your demo PDF. Mark 4 to 5 of them `answerable = no` (the PDF cannot answer them). Include several Tamil questions.
2. Run: `python eval.py your_document.pdf questions.csv --provider gemini`
3. Open `results.csv`, correct any wrong automatic verdicts by hand, and re-count.
4. Use `results_chart.png` and the printed numbers on your slide. Quote only your own measured numbers.

## Suggested demo flow (about 2 minutes)
1. Upload the PDF and point at the pages and chunks counters.
2. Ask an English question and open Source 1 to show the highlighted page.
3. Ask a Tamil question about the same fact.
4. Ask a question the document cannot answer and let PinPoint say "Not found in the document."

## Tuning
- Answers wrongly refused: lower the relevance threshold in the sidebar.
- Off-topic questions still answered: raise the threshold.
- Answers cut off mid-idea: increase chunk size.

## Limitations (say these honestly in your presentation)
- Scanned or image-only PDFs have no selectable text and are not supported (no OCR).
- Highlighting is best effort and can miss text in complex layouts or tables.
- Retrieval quality for low-resource languages is lower than for English.
- Answers depend on the LLM following the grounding instructions; the threshold and source display are the safeguards.
