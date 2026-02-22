"""
Medical RAG Chatbot — Chainlit UI
Powered by: MedQuAD · intfloat/e5-base · Llama 3.1 via Groq
"""

import os
from dotenv import load_dotenv

load_dotenv()

import chainlit as cl
from chainlit.input_widget import Slider, Switch
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "MedQuAD_combined.csv"
EMBEDDINGS_CACHE = "embeddings_cache.npy"
EMBED_MODEL_NAME = "intfloat/e5-base"
GROQ_MODEL = "llama-3.1-8b-instant"
DEFAULT_K = 3
DEFAULT_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Module-level globals — loaded once, shared across all sessions
# ---------------------------------------------------------------------------
_model = None
_df = None
_embeddings = None
_index = None
_llm = None


def _load_pipeline():
    global _model, _df, _embeddings, _index
    if _model is not None:
        return _model, _df, _embeddings, _index

    _model = SentenceTransformer(EMBED_MODEL_NAME)

    _df = pd.read_csv(DATA_PATH)
    _df = _df.copy()
    _df["answer"] = (
        _df["answer"]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"  +", " ", regex=True)
        .str.strip()
    )
    _df["text"] = (
        "Focus Area: " + _df["focus"].fillna("") +
        " ; Question: " + _df["question"].fillna("") +
        " ; Question Type: " + _df["question_qtype"].fillna("") +
        " ; Source: " + _df["url"].fillna("") +
        " ; Answer: " + _df["answer"].fillna("")
    )

    n_rows = len(_df)
    cache_valid = (
        os.path.exists(EMBEDDINGS_CACHE)
        and np.load(EMBEDDINGS_CACHE, mmap_mode="r").shape[0] == n_rows
    )
    if cache_valid:
        _embeddings = np.load(EMBEDDINGS_CACHE)
    else:
        texts = _df["text"].tolist()
        _embeddings = _model.encode(
            texts,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        np.save(EMBEDDINGS_CACHE, _embeddings)

    dimension = _embeddings.shape[1]
    _index = faiss.IndexFlatIP(dimension)
    _index.add(_embeddings)

    return _model, _df, _embeddings, _index


def _load_llm():
    global _llm
    if _llm is not None:
        return _llm

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return None

    try:
        _llm = ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0.0)
        return _llm
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def retrieve(query, model, df, embeddings, index, k, threshold):
    query_emb = model.encode(
        ["query: " + query], convert_to_numpy=True, normalize_embeddings=True
    )
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = sims.argsort()[::-1][:k]
    best_score = float(sims[top_idx[0]])

    if best_score < threshold:
        return None, best_score

    results = []
    for rank, idx in enumerate(top_idx, start=1):
        results.append({
            "rank": rank,
            "score": float(sims[idx]),
            "focus": df.iloc[idx]["focus"],
            "question": df.iloc[idx]["question"],
            "url": df.iloc[idx]["url"],
            "answer": df.iloc[idx]["answer"].strip(),
            "text": df.iloc[idx]["text"],
        })
    return results, best_score


def build_prompt(query, retrieved_docs):
    context_parts = [
        doc.get("text", str(doc)) if isinstance(doc, dict) else str(doc)
        for doc in retrieved_docs
    ]
    context_text = "\n\n---\n\n".join(context_parts)
    return f"""You are an expert medical assistant. Answer the user's question based SOLELY \
on the provided context. If the context does not contain the answer, clearly state that \
the information is not available in the provided documents.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""


REJECTION_PHRASES = [
    "cannot answer that",
    "i don't see any context",
    "no information provided",
    "as a large language model",
    "outside the focus area",
]


def _fallback_answer(retrieved_docs):
    top = retrieved_docs[0]
    result = f"**{top['focus']}**\n\n{top['answer']}"
    url = top.get("url", "")
    if url.startswith("http"):
        result += f"\n\n**Source:** {url}"
    result += "\n\n*Note: AI synthesis unavailable — showing best-matched knowledge base entry.*"
    return result


async def _generate_streaming(query, retrieved_docs, llm, msg):
    """Stream LLM tokens into a Chainlit message; returns the full answer text."""
    if llm is None:
        return _fallback_answer(retrieved_docs)

    prompt = build_prompt(query, retrieved_docs)
    full_answer = ""

    try:
        async for chunk in llm.astream(prompt):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            await msg.stream_token(token)
            full_answer += token

        full_answer = full_answer.strip()

        if not full_answer:
            return _fallback_answer(retrieved_docs)

        for phrase in REJECTION_PHRASES:
            if phrase in full_answer.lower():
                return (
                    "The provided context does not contain information relevant "
                    "to your question. Please ask a medical question covered by "
                    "the MedQuAD knowledge base."
                )

        url = retrieved_docs[0].get("url", "") if retrieved_docs else ""
        if url.startswith("http"):
            full_answer += f"\n\n**Source:** {url}"

        return full_answer

    except Exception:
        return _fallback_answer(retrieved_docs)


# ---------------------------------------------------------------------------
# Chainlit lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    loading = cl.Message(content="⏳ Loading model & building index...")
    await loading.send()

    model, df, embeddings, index = _load_pipeline()
    llm = _load_llm()

    # Persist defaults in user session
    cl.user_session.set("k_docs", DEFAULT_K)
    cl.user_session.set("threshold", DEFAULT_THRESHOLD)
    cl.user_session.set("show_sources", True)

    # Sidebar settings panel
    await cl.ChatSettings([
        Slider(
            id="k_docs",
            label="Top-k documents to retrieve",
            initial=DEFAULT_K,
            min=1,
            max=10,
            step=1,
        ),
        Slider(
            id="threshold",
            label="Similarity threshold",
            initial=DEFAULT_THRESHOLD,
            min=0.0,
            max=1.0,
            step=0.05,
        ),
        Switch(id="show_sources", label="Show retrieved sources", initial=True),
    ]).send()

    num_pairs = len(df)
    num_cats = df["folder_name"].nunique()
    status = (
        f"✅ Ready — **{num_pairs:,}** medical Q&A pairs indexed "
        f"across **{num_cats}** source categories."
    )
    if llm is None:
        status += (
            "\n\n⚠️ `GROQ_API_KEY` not found. "
            "Add it to your `.env` file. "
            "Answers will fall back to the best-matched knowledge base entry."
        )

    loading.content = status
    await loading.update()


@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("k_docs", int(settings["k_docs"]))
    cl.user_session.set("threshold", float(settings["threshold"]))
    cl.user_session.set("show_sources", bool(settings["show_sources"]))


@cl.on_message
async def on_message(message: cl.Message):
    model, df, embeddings, index = _load_pipeline()
    llm = _load_llm()

    k_docs = cl.user_session.get("k_docs") or DEFAULT_K
    threshold = cl.user_session.get("threshold") or DEFAULT_THRESHOLD
    show_sources = cl.user_session.get("show_sources")
    if show_sources is None:
        show_sources = True

    query = message.content

    results, best_score = retrieve(
        query, model, df, embeddings, index, k=k_docs, threshold=threshold
    )

    if results is None:
        await cl.Message(
            content=(
                f"No relevant medical information found "
                f"(best similarity: **{best_score:.2f}**, threshold: **{threshold}**). "
                "Try a more specific question, or lower the similarity threshold in Settings."
            )
        ).send()
        return

    # Stream the answer
    answer_msg = cl.Message(content="")
    await answer_msg.send()
    final_answer = await _generate_streaming(query, results, llm, answer_msg)

    # If fallback/rejection replaced the streamed content, update the message
    if final_answer != answer_msg.content:
        answer_msg.content = final_answer
        await answer_msg.update()

    # Show source citations as a follow-up message
    if show_sources and results:
        sources_md = "### Retrieved Sources\n\n"
        for src in results:
            snippet = src["answer"][:400]
            if len(src["answer"]) > 400:
                snippet += "..."
            sources_md += (
                f"**Rank {src['rank']}** · Score: `{src['score']:.4f}` "
                f"· Focus: *{src['focus']}*\n\n"
                f"> {snippet}\n\n"
                f"[View Source]({src['url']})\n\n---\n\n"
            )
        await cl.Message(content=sources_md, author="Sources").send()
