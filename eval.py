"""
Evaluate PinPoint (RAG) against a plain LLM that has no document access.

Usage:
    python eval.py your_document.pdf questions.csv --provider claude

questions.csv columns:
    question    the question text (any language)
    expected    one or more keywords that a correct answer must contain, separated by |
                (leave blank for questions the document cannot answer)
    answerable  yes / no   (use "no" for questions whose answer is NOT in the document)

Outputs: results.csv (every answer, for manual review) and results_chart.png (for your slide).
Keyword matching is a rough automatic grader: open results.csv and fix any wrong verdicts by hand
before you quote numbers on a slide.
"""
import argparse

import pandas as pd

from pinpoint_core import (
    NOT_FOUND,
    Retriever,
    answer_question,
    chunk_pages,
    detect_provider,
    load_embedder,
    load_pdf_pages,
    plain_llm_answer,
)

DECLINE_HINTS = ("not found", "don't know", "do not know", "no information", "cannot find", "can't find", "not sure", "unable to")


def is_decline(text):
    t = text.lower()
    return any(h in t for h in DECLINE_HINTS)


def grade(answerable, expected, answer):
    """Return (correct, hallucinated)."""
    if answerable == "yes":
        keys = [k.strip().lower() for k in str(expected).split("|") if k.strip()]
        correct = any(k in answer.lower() for k in keys)
        return correct, (not correct and not is_decline(answer))
    declined = is_decline(answer)
    return declined, (not declined)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("questions")
    ap.add_argument("--provider", default=detect_provider(), choices=["claude", "gemini"])
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    pdf_bytes = open(args.pdf, "rb").read()
    chunks = chunk_pages(load_pdf_pages(pdf_bytes))
    retriever = Retriever(load_embedder(), chunks)
    qs = pd.read_csv(args.questions).fillna("")

    rows = []
    for _, r in qs.iterrows():
        q, exp, ans_flag = r["question"], r["expected"], str(r["answerable"]).strip().lower()
        rag = answer_question(q, retriever, k=args.k, min_score=args.min_score, provider=args.provider)
        plain = plain_llm_answer(q, args.provider)
        rag_ok, rag_h = grade(ans_flag, exp, rag["answer"])
        pl_ok, pl_h = grade(ans_flag, exp, plain)
        rows.append(
            dict(question=q, answerable=ans_flag, expected=exp, rag_answer=rag["answer"], plain_answer=plain,
                 rag_correct=rag_ok, rag_hallucinated=rag_h, plain_correct=pl_ok, plain_hallucinated=pl_h)
        )
        print(f"RAG {'OK ' if rag_ok else 'BAD'} | PLAIN {'OK ' if pl_ok else 'BAD'} | {q[:70]}")

    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False, encoding="utf-8-sig")
    n = len(df)
    summary = {
        "RAG (PinPoint)": (df.rag_correct.mean() * 100, df.rag_hallucinated.sum()),
        "Plain LLM": (df.plain_correct.mean() * 100, df.plain_hallucinated.sum()),
    }
    print(f"\nQuestions: {n}")
    for name, (acc, hal) in summary.items():
        print(f"{name:16s} accuracy {acc:5.1f}%   hallucinated answers: {hal}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    names = list(summary)
    ax[0].bar(names, [summary[k][0] for k in names], color=["#0F766E", "#94A3B8"])
    ax[0].set_ylim(0, 100)
    ax[0].set_title("Accuracy (%)")
    ax[1].bar(names, [int(summary[k][1]) for k in names], color=["#0F766E", "#94A3B8"])
    ax[1].set_title("Hallucinated answers (count)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("results_chart.png", dpi=200)
    print("Saved results.csv and results_chart.png")


if __name__ == "__main__":
    main()
