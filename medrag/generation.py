"""Prompting and answer generation, with a fallback when the LLM is unavailable."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

REJECTION_PHRASES = [
    "cannot answer that",
    "i don't see any context",
    "no information provided",
    "as a large language model",
    "outside the focus area",
    "provided context does not contain information",
    "not available in the provided documents",
]

NOT_COVERED_MESSAGE = (
    "The provided context does not contain information relevant to your question. "
    "Please ask a medical question covered by the MedQuAD knowledge base."
)


def build_prompt(query: str, retrieved_docs: list[dict]) -> str:
    context_text = "\n\n---\n\n".join(doc["text"] for doc in retrieved_docs)
    return f"""You are an expert medical assistant. Answer the user's question based SOLELY \
on the provided context. If the context does not contain the answer, clearly state that \
the information is not available in the provided documents.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""


def format_sources(retrieved_docs: list[dict]) -> str:
    """List each distinct source once, because the answer may draw on any retrieved passage."""
    urls = list(dict.fromkeys(doc["url"] for doc in retrieved_docs))
    return "\n\n**Sources:**\n" + "\n".join(f"- {url}" for url in urls)


def fallback_answer(retrieved_docs: list[dict]) -> str:
    """Return the top retrieved answer when the LLM is unavailable."""
    top = retrieved_docs[0]
    return (
        f"**{top['focus']}**\n\n{top['answer']}\n\n**Source:** {top['url']}"
        "\n\n*Note: AI synthesis unavailable — showing best-matched knowledge base entry.*"
    )


def generate_answer(query: str, retrieved_docs: list[dict], llm: Any) -> str:
    if llm is None:
        return fallback_answer(retrieved_docs)
    try:
        response = llm.invoke(build_prompt(query, retrieved_docs))
    except Exception:  # network, quota, auth or retired model: still show the sourced passage
        logger.exception("LLM call failed; falling back to the retrieved passage")
        return fallback_answer(retrieved_docs)

    answer = str(getattr(response, "content", response)).strip()
    if not answer:
        return fallback_answer(retrieved_docs)
    if any(phrase in answer.lower() for phrase in REJECTION_PHRASES):
        return NOT_COVERED_MESSAGE
    return answer + format_sources(retrieved_docs)
