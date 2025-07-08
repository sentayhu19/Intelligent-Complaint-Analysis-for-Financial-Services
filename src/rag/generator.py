from typing import Dict, Any, List
import os
from huggingface_hub import InferenceApi

from src.rag.prompt import TEMPLATE
from src.rag.retriever import Retriever


class RAGPipeline:
    """High-level retrieval-augmented generation pipeline."""

    def __init__(self, top_k: int = 5, model_name: str = "google/flan-t5-base"):
        self.retriever = Retriever(top_k=top_k)
        # Use hosted inference API – avoids local download. Requires HF_TOKEN env for rate limits.
        token = os.getenv("HF_TOKEN")
        self.client = InferenceApi(repo_id=model_name, token=token)

    def answer(self, question: str) -> Dict[str, Any]:
        hits = self.retriever.retrieve(question)
        context = self.retriever.format_context(hits)
        prompt = TEMPLATE.format(context=context, question=question)
        raw = self.client(inputs=prompt, params={"max_new_tokens": 256}, raw_response=True)
        resp = raw.json()
        # API may return dict or list depending on model
        if isinstance(resp, list):
            answer = resp[0]["generated_text"]
        elif isinstance(resp, dict):
            answer = resp.get("generated_text", "")
        else:
            answer = str(resp)
        # Strip prompt if echoed back
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        return {"question": question, "answer": answer, "sources": hits}
