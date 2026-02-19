# Medical RAG Chatbot

An **end-to-end Retrieval-Augmented Generation (RAG) chatbot** for medical question answering, powered by the **MedQuAD** dataset (16,407 Q&A pairs), **Mistral** via Ollama, and a **Streamlit** web UI.

---

## Project Overview

Large Language Models often hallucinate medical information. This project addresses that by implementing a **RAG architecture** — relevant Q&A pairs are retrieved from the MedQuAD knowledge base first, then injected into the prompt before Mistral generates a response.

---

## Features

- 16,407 medical Q&A pairs from the [MedQuAD](https://github.com/abachaa/MedQuAD) dataset
- Semantic search with `intfloat/e5-base` embeddings + FAISS index
- Response generation with **Mistral** running locally via Ollama
- Embedding cache — computed once, reused on every subsequent launch
- Streamlit chat UI with configurable top-k retrieval and similarity threshold
- Retrieved sources shown inline with each answer

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Mistral (via Ollama) |
| Embeddings | `intfloat/e5-base` (Sentence Transformers) |
| Vector DB | FAISS |
| Orchestration | LangChain |
| Dataset | MedQuAD (Excel → CSV) |
| Web UI | Streamlit |
| Language | Python 3.10+ |

---

## Installation & Setup

### 1. Clone the repo

```bash
git clone https://github.com/dipdhru/medical_chatbot_rag.git
cd medical_chatbot_rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama and pull Mistral

```bash
# Install Ollama: https://ollama.com
ollama serve                  # start the Ollama server (keep running)
ollama pull mistral           # download Mistral (~4 GB)
```

### 4. Convert the dataset to CSV

The app expects `MedQuAD_combined.csv` in the project root:

```bash
python3 -c "
import pandas as pd
df = pd.read_excel('Data Set/MedQuAD_combined.xlsx')
df.to_csv('MedQuAD_combined.csv', index=False)
print(f'Saved {len(df):,} rows')
"
```

---

## Running the App

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> **First launch:** embeddings for all 16k records are computed and saved to `embeddings_cache.npy`. This takes a few minutes. Subsequent launches are instant.

---

## Project Structure

```
medical_chatbot_rag/
├── app.py                    # Streamlit chat UI
├── RAG_llama.ipynb           # Experimental notebook
├── requirements.txt          # Python dependencies
├── Data Set/
│   └── MedQuAD_combined.xlsx # Source dataset
├── MedQuAD_combined.csv      # Generated — create via step 4 above
└── embeddings_cache.npy      # Generated — created on first app launch
```

---

## How It Works

1. **Query** — user submits a medical question
2. **Embed** — query is encoded with `intfloat/e5-base`
3. **Retrieve** — FAISS finds the top-k most similar Q&A pairs by cosine similarity
4. **Filter** — results below the similarity threshold are discarded
5. **Generate** — Mistral produces an answer grounded in the retrieved context
6. **Cite** — the primary source URL is appended to the answer
