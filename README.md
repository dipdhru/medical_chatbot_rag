---
title: Medical RAG Chatbot
sdk: streamlit
emoji: 🏥
colorFrom: blue
colorTo: green
pinned: false
---

# Medical RAG Chatbot

An **end-to-end Retrieval-Augmented Generation (RAG) chatbot** for medical question answering, powered by the **MedQuAD** dataset (16,407 Q&A pairs), **Llama 3.1 8B** via Groq, and a **Streamlit** web UI.

**[Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/dipdhrusingh/medical-chatbot)**

---

## Why RAG?

Large Language Models frequently hallucinate medical facts. RAG solves this by **grounding every answer in verified source documents** — the LLM can only use information retrieved from a curated medical knowledge base, and every response includes a source citation.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Llama 3.1 8B (via Groq API) |
| Embeddings | `intfloat/e5-base` (Sentence Transformers) |
| Vector Search | FAISS (IndexFlatIP, cosine similarity) |
| Dataset | MedQuAD — 16,407 medical Q&A pairs |
| Web UI | Streamlit |
| Evaluation | ROUGE, BLEU, BERTScore, RAGAS |
| Deployment | Hugging Face Spaces |

---

## How It Works

```
User Question
      |
      v
 [ Embed ]      intfloat/e5-base (768-dim)
      |
      v
 [ Retrieve ]   FAISS top-k + cosine similarity
      |
      v
 [ Filter ]     similarity threshold + source URL validation
      |
      v
 [ Generate ]   Llama 3.1 8B via Groq (grounded prompt)
      |
      v
 [ Cite ]       append verified source URL
```

---

## Project Structure

```
medical_chatbot_rag/
├── app.py                    # Streamlit web UI (production)
├── RAG_llama.ipynb           # Full pipeline notebook with evaluation
├── MedQuAD_combined.csv      # MedQuAD dataset (16,407 Q&A pairs)
├── embeddings_cache.npy      # Pre-computed embeddings (auto-generated)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .devcontainer/            # VS Code / Codespaces config
│   └── devcontainer.json
└── .gitattributes            # Git LFS tracking rules
```

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/dipdhru/medical_chatbot_rag.git
cd medical_chatbot_rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your Groq API key

Get a free key at [console.groq.com](https://console.groq.com), then:

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

### 4. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

> **First launch:** embeddings for all 16k records are computed and cached to `embeddings_cache.npy`. This takes a few minutes. Subsequent launches are instant.

---

## Features

- **16,407 medical Q&A pairs** from the MedQuAD dataset
- **Semantic search** with e5-base embeddings + FAISS
- **Source-grounded answers** — every response includes a verifiable source URL
- **Out-of-domain rejection** — non-medical queries are filtered by similarity threshold
- **Configurable settings** — adjust top-k retrieval and similarity threshold via sidebar
- **Graceful fallback** — works without API key (shows best-matched knowledge base entry)

---

## Evaluation Results

See `RAG_llama.ipynb` for the full evaluation pipeline. Metrics include:

- **Retrieval**: Precision@3, Recall@3, MRR@3
- **Generation**: ROUGE-L, BLEU-4, BERTScore
- **RAG Quality**: RAGAS (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
