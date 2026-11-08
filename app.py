"""Medical RAG Chatbot: Streamlit UI.

Powered by MedQuAD, intfloat/e5-base embeddings and GPT-OSS 20B via Groq.
"""

import os

import numpy as np
import streamlit as st
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

from medrag.config import (
    DATA_PATH,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
    EMBED_MODEL_NAME,
    EMBEDDINGS_CACHE,
    GROQ_MODEL,
)
from medrag.data import load_knowledge_base
from medrag.generation import generate_answer
from medrag.retrieval import Retriever


@st.cache_resource(show_spinner=False)
def load_pipeline() -> Retriever:
    """Load the model, knowledge base and embeddings once per process."""
    model = SentenceTransformer(EMBED_MODEL_NAME)
    documents = load_knowledge_base(DATA_PATH)
    try:
        embeddings = np.load(EMBEDDINGS_CACHE, allow_pickle=False)
        cache_valid = embeddings.shape[0] == len(documents)
    except (OSError, ValueError):
        cache_valid = False

    if not cache_valid:
        embeddings = model.encode(
            documents["text"].tolist(),
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        np.save(EMBEDDINGS_CACHE, embeddings)

    return Retriever(model, documents, embeddings)


def groq_api_key() -> str | None:
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:  # Streamlit raises when no secrets file exists
        key = None
    return key or os.environ.get("GROQ_API_KEY")


@st.cache_resource(show_spinner=False)
def load_llm():
    """Create the Groq client; returns None if the key is missing or invalid."""
    api_key = groq_api_key()
    if not api_key:
        return None
    try:
        return ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0.0)
    except Exception:
        return None


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
    "**intfloat/e5-base** embeddings · **GPT-OSS 20B** via Groq"
)

st.warning(
    "For information only. This is a portfolio project, not medical advice. "
    "For anything about your health, talk to a qualified clinician, and call your local "
    "emergency number in an emergency.",
    icon="⚕️",
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
        step=0.01,
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
        "- `GROQ_API_KEY` in secrets or env\n"
        "- Free key at console.groq.com\n"
        "- `MedQuAD_combined.csv` in the same directory"
    )

# ── Load pipeline ──────────────────────────────────────────────────────────
with st.spinner(
    "Loading model & building index… "
    "(first run computes embeddings — this may take a few minutes)"
):
    retriever = load_pipeline()
    df = retriever.documents
    llm = load_llm()

if llm is None:
    st.warning(
        "Groq API key not found. "
        "Add **GROQ_API_KEY** to your Streamlit secrets (`.streamlit/secrets.toml`) "
        "or as an environment variable. Get a free key at https://console.groq.com. "
        "Answers will fall back to the best-matched knowledge base entry."
    )

st.success(
    f"Ready — **{len(df):,}** medical Q&A pairs indexed across "
    f"**{df['folder_name'].nunique()}** source categories."
)

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
            results, best_score = retriever.retrieve(prompt, k=k_docs, threshold=threshold)

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
