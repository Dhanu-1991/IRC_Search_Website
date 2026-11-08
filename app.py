

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.ingestion_pipeline.ingest import main as ingest_documents
from src.retrivel_pipeline.Groq_llm import GroqLLM
from src.retrivel_pipeline.Retrieval import RetrievalEngine
from src.retrivel_pipeline.generation import format_context, PROMPT_TEMPLATE


ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

st.set_page_config(
    page_title="IRC Search Assistant",
    page_icon="IRC",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the IRC search engine...")
def get_retriever() -> RetrievalEngine:
    return RetrievalEngine()


def answer_with_sources(question: str, top_k: int) -> tuple[str, list[dict]]:
    retriever = get_retriever()
    results = retriever.retrieve(question, top_k=top_k)
    if not results:
        return "I couldn't find any relevant information in the IRC documents.", []

    prompt = PROMPT_TEMPLATE.format(
        context=format_context(results),
        question=question,
    )
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    answer = GroqLLM(model_name=model_name).generate_response(prompt)
    return answer, results


def render_source(result: dict, index: int) -> None:
    metadata = result.get("metadata") or {}
    source = (
        metadata.get("source")
        or metadata.get("file_path")
        or metadata.get("filename")
        or "Unknown document"
    )
    page = metadata.get("page")
    page_label = f" - page {int(page) + 1}" if isinstance(page, (int, float)) else ""
    distance = result.get("distance")
    score_label = f"{distance:.3f}" if isinstance(distance, (int, float)) else "n/a"

    with st.expander(f"{index}. {Path(str(source)).name}{page_label} - distance {score_label}"):
        st.caption(str(source))
        st.write(result.get("content", ""))


def main() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("IRC Search Assistant")
    st.write("Ask questions about Indian Roads Congress standards using your indexed PDF documents.")

    with st.sidebar:
        st.header("Controls")
        top_k = st.slider("Source chunks", min_value=1, max_value=10, value=5)

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.header("Add IRC PDFs")
        uploads = st.file_uploader(
            "Upload one or more PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if uploads and st.button("Index uploaded documents", use_container_width=True):
            data_dir = ROOT_DIR / "data"
            data_dir.mkdir(exist_ok=True)
            for upload in uploads:
                (data_dir / upload.name).write_bytes(upload.getvalue())

            with st.spinner("Extracting text and creating embeddings..."):
                # Rebuild from every PDF currently in data/ so replaced files
                # cannot leave stale chunks in the collection.
                ingest_documents(str(data_dir), reset=True)
                get_retriever.clear()
            st.session_state.messages = []
            st.session_state.index_message = (
                f"Indexed all PDFs currently stored ({len(list(data_dir.glob('*.pdf')))} document(s))."
            )
            st.rerun()

        if st.session_state.get("index_message"):
            st.success(st.session_state.pop("index_message"))

        if not os.getenv("GROQ_API_KEY"):
            st.warning("Add GROQ_API_KEY to a .env file before asking a question.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                sources = message.get("sources", [])
                if sources:
                    st.markdown("#### Sources")
                    for index, result in enumerate(sources, start=1):
                        render_source(result, index)

    question = st.chat_input("Ask about an IRC standard, clause, or requirement...")
    if not question or not question.strip():
        return

    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the IRC documents..."):
            try:
                answer, sources = answer_with_sources(question, top_k)
            except Exception as error:
                st.error(f"Could not answer the question: {error}")
                return

        st.markdown(answer)
        if sources:
            st.markdown("#### Sources")
            for index, result in enumerate(sources, start=1):
                render_source(result, index)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


if __name__ == "__main__":
    main()
