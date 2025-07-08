"""Streamlit app for interacting with the RAG pipeline (Task-4)."""

import streamlit as st
from typing import List, Dict, Any

from src.rag.generator import RAGPipeline

st.set_page_config(page_title="Complaint RAG Chat", page_icon="💬", layout="wide")

# --- Initialize pipeline & session state ---
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline(top_k=5)
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []  # [{q,a,sources}]

# --- Sidebar ---
with st.sidebar:
    st.title("💡 How to use")
    st.markdown(
        "Type a question about customer complaints and press *Ask*. "
        "The answer and its supporting excerpts will appear below."
    )
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.experimental_rerun()

# --- Main Interface ---
st.title("📞 Consumer Complaint Q&A Chatbot")
question = st.text_input("Your question:", key="user_input", placeholder="e.g. What issues do consumers report about overdraft fees?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked and question.strip():
    with st.spinner("Retrieving & generating answer..."):
        result = st.session_state.rag.answer(question)
    st.session_state.history.insert(0, result)  # newest first

# --- Display history ---
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        st.markdown(item["answer"])
        with st.expander("Sources"):
            for i, src in enumerate(item["sources"], 1):
                st.markdown(f"**Source {i} (score {src.get('score', 0):.2f}):**")
                st.write(src.get("text", ""))
