from typing import Dict, Any, List
from transformers import pipeline

from src.rag.prompt import TEMPLATE
from src.rag.retriever import Retriever


class RAGPipeline:
    """High-level retrieval-augmented generation pipeline."""

    def __init__(self, top_k: int = 5, model_name: str = "google/flan-t5-base", device: int = -1):
        self.retriever = Retriever(top_k=top_k)
        # Use text-generation pipeline (can fallback to smaller model if GPU not available)
        self.llm = pipeline(
            "text2text-generation",
            model=model_name,
            device=device,
            max_new_tokens=256,
            do_sample=False,
        )

    def answer(self, question: str) -> Dict[str, Any]:
        hits = self.retriever.retrieve(question)
        context = self.retriever.format_context(hits)
        prompt = TEMPLATE.format(context=context, question=question)
        raw = self.llm(prompt)[0]["generated_text"]
        # Everything after last "Answer:" token
        answer = raw.split("Answer:")[-1].strip()
        return {"question": question, "answer": answer, "sources": hits}
