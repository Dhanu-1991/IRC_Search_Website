
from typing import List, Dict, Any

from src.retrivel_pipeline.Groq_llm import GroqLLM
from src.retrivel_pipeline.Retrieval import RetrievalEngine


PROMPT_TEMPLATE = """
You are an expert on Indian Roads Congress (IRC) standards.

Answer the user's question using ONLY the information provided below.
If you use any fact, cite its source using [Page N] when a page number is
available in the metadata. Do not mention filenames or file extensions.

If the provided context does not contain enough information to answer,
clearly state that you could not find the required information.

--- CONTEXT ---
{context}
--- END CONTEXT ---

Question: {question}

Answer:
"""


def format_context(results: List[Dict[str, Any]]) -> str:
    """Format retrieved document chunks into a single context string."""

    context_parts = []

    for result in results:
        metadata = result["metadata"]

        page = metadata.get("page")
        source = f"Page {int(page) + 1}" if isinstance(page, (int, float)) else "Relevant section"

        content = result["content"].replace("\n", " ").strip()

        context_parts.append(f"[{source}]: {content}")

    return "\n".join(context_parts)


def answer_question( question: str, retriever: RetrievalEngine, top_k: int = 5, llm_model: str = "openai/gpt-oss-120b") -> str:
    """Retrieve relevant documents and generate an answer using Groq."""

    # Retrieve relevant document chunks
    results = retriever.retrieve(question, top_k=top_k)

    if not results:
        return "I couldn't find any relevant information in the IRC documents."

    # Prepare the context and prompt
    context = format_context(results)
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    # Generate the answer using the Groq LLM
    groq = GroqLLM(model_name=llm_model)

    answer = groq.generate_response(prompt)

    return answer
