"""Simple qualitative evaluation runner producing markdown table."""

from typing import List
from tabulate import tabulate

from src.rag.generator import RAGPipeline

QUESTIONS: List[str] = [
    "What are the main issues customers have with credit card billing?",
    "How long do mortgage-related complaints usually take to resolve?",
    "Do consumers frequently report identity theft problems?",
    "Which states see the most loan servicing complaints?",
    "What are common sentiments about overdraft fees?",
]

def run_eval():
    rag = RAGPipeline()
    rows = []
    for q in QUESTIONS:
        result = rag.answer(q)
        top_src = result["sources"][0]["text"][:120].replace("\n", " ")+" …" if result["sources"] else "-"
        rows.append([
            q,
            result["answer"][:240] + ("…" if len(result["answer"]) > 240 else ""),
            top_src,
            "",  # Score placeholder
            "",  # Comments placeholder
        ])
    print(tabulate(rows, headers=[
        "Question", "Generated Answer", "Top Source Snippet", "Score (1-5)", "Comments"], tablefmt="github"))

if __name__ == "__main__":
    run_eval()
