"""
Medical RAG Chatbot — Streamlit UI
Powered by: MedQuAD · intfloat/e5-base · Mistral via Ollama
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = "MedQuAD_combined.csv"
EMBEDDINGS_CACHE = "embeddings_cache.npy"
EMBED_MODEL_NAME = "intfloat/e5-base"
OLLAMA_MODEL = "mistral:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_K = 3
DEFAULT_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Cached resource loaders  (run once per process, cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load embedding model, data, embeddings (from cache or fresh), and FAISS index."""

    # 1. Embedding model
    model = SentenceTransformer(EMBED_MODEL_NAME)

    # 2. Dataset
    df = pd.read_csv(DATA_PATH)
    df = df.copy()
    df["answer"] = (
        df["answer"]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"  +", " ", regex=True)
        .str.strip()
    )
    df["text"] = (
        "Focus Area: " + df["focus"].fillna("") +
        " ; Question: " + df["question"].fillna("") +
        " ; Question Type: " + df["question_qtype"].fillna("") +
        " ; Source: " + df["url"].fillna("") +
        " ; Answer: " + df["answer"].fillna("")
    )

    # 3. Embeddings — load from cache or compute and save
    if os.path.exists(EMBEDDINGS_CACHE):
        embeddings = np.load(EMBEDDINGS_CACHE)
    else:
        texts = df["text"].tolist()
        batch_size = 16
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            batches.append(emb)
        embeddings = np.vstack(batches).astype("float32")
        np.save(EMBEDDINGS_CACHE, embeddings)

    # 4. FAISS index (inner-product on normalised vectors == cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return model, df, embeddings, index


@st.cache_resource(show_spinner=False)
def load_llm():
    """Initialise the local Ollama LLM client."""
    try:
        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RAG helpers
# ---------------------------------------------------------------------------

def retrieve(query: str, model, df, embeddings, index, k: int, threshold: float):
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
        results.append(
            {
                "rank": rank,
                "score": float(sims[idx]),
                "focus": df.iloc[idx]["focus"],
                "question": df.iloc[idx]["question"],
                "url": df.iloc[idx]["url"],
                "answer": df.iloc[idx]["answer"].strip(),
                "text": df.iloc[idx]["text"],
            }
        )
    return results, best_score


def build_prompt(query: str, retrieved_docs: list) -> str:
    context_parts = []
    for doc in retrieved_docs:
        context_parts.append(doc.get("text", str(doc)) if isinstance(doc, dict) else str(doc))
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


def generate_answer(query: str, retrieved_docs: list, llm) -> str:
    if llm is None:
        return (
            "Ollama LLM is not reachable. "
            "Make sure Ollama is running and `mistral:latest` is pulled."
        )

    prompt = build_prompt(query, retrieved_docs)
    try:
        response = llm.invoke(prompt)
        # ChatOllama returns an AIMessage; plain Ollama LLM returns a str
        ans = response.content if hasattr(response, "content") else str(response)
        ans = ans.strip()

        if not ans:
            return "I could not generate an answer. Please try rephrasing your question."

        for phrase in REJECTION_PHRASES:
            if phrase in ans.lower():
                return (
                    "The provided context does not contain information relevant "
                    "to your question. Please ask a medical question covered by "
                    "the MedQuAD knowledge base."
                )

        # Append primary source URL
        source_url = retrieved_docs[0].get("url", "") if retrieved_docs else ""
        if source_url.startswith("http"):
            ans += f"\n\n**Source:** {source_url}"

        return ans

    except Exception as exc:
        return f"An error occurred during generation: {exc}"


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Medical RAG Chatbot")
st.caption(
    "Powered by **MedQuAD** · "
    "**intfloat/e5-base** embeddings · **Mistral** via Ollama"
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    k_docs = st.slider(
        "Top-k documents to retrieve", min_value=1, max_value=10, value=DEFAULT_K
    )
    threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help="Queries below this cosine-similarity score return 'no results found'.",
    )
    show_sources = st.toggle("Show retrieved sources", value=True)

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.info(
        "**Requirements**\n"
        "- Ollama running locally\n"
        "- `ollama pull mistral`\n"
        "- `MedQuAD_combined.csv` in the same directory"
    )

# ── Load pipeline ──────────────────────────────────────────────────────────
with st.spinner(
    "Loading model & building index… "
    "(first run computes embeddings — this may take a few minutes)"
):
    model, df, embeddings, index = load_pipeline()
    llm = load_llm()

if llm is None:
    st.warning(
        "Ollama LLM not reachable. "
        "Start Ollama (`ollama serve`) and pull the model (`ollama pull mistral`) "
        "before asking questions."
    )

st.success(f"Ready — **{len(df):,}** medical Q&A pairs indexed across **{df['folder_name'].nunique()}** source categories.")

# ── Chat history ───────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if show_sources and message.get("sources"):
            with st.expander("📄 Retrieved sources"):
                for src in message["sources"]:
                    st.markdown(
                        f"**Rank {src['rank']}** · Score: `{src['score']:.4f}` · "
                        f"Focus: *{src['focus']}*"
                    )
                    st.markdown(f"> {src['answer'][:400]}...")
                    st.caption(src["url"])

# ── Chat input ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a medical question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base…"):
            results, best_score = retrieve(
                prompt, model, df, embeddings, index, k=k_docs, threshold=threshold
            )

        if results is None:
            answer = (
                f"No relevant medical information found for your query "
                f"(best similarity score: **{best_score:.2f}**, threshold: **{threshold}**). "
                "Try a more specific medical question, or lower the similarity threshold."
            )
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": []}
            )
        else:
            with st.spinner("Generating answer…"):
                answer = generate_answer(prompt, results, llm)

            st.markdown(answer)

            if show_sources:
                with st.expander("📄 Retrieved sources"):
                    for src in results:
                        st.markdown(
                            f"**Rank {src['rank']}** · Score: `{src['score']:.4f}` · "
                            f"Focus: *{src['focus']}*"
                        )
                        st.markdown(f"> {src['answer'][:400]}...")
                        st.caption(src["url"])

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": results}
            )
