# Retrieval-Augmented Generation (RAG) System using LLaMA

An **end-to-end Retrieval-Augmented Generation (RAG) pipeline** implemented in Python using **LLaMA-based Large Language Models (LLMs)**.  
This project demonstrates how to combine **semantic search, vector databases, and transformer models** to generate accurate, context-aware responses from private document collections.

---

## 📌 Project Overview

Large Language Models often hallucinate when answering questions without access to source data.  
This project addresses that problem by implementing a **RAG architecture**, where relevant documents are retrieved first and injected into the LLM prompt before response generation.

**Primary focus areas:**
- Natural Language Processing (NLP)
- Information Retrieval
- Large Language Models (LLMs)
- Vector Search & Embeddings
- Applied Machine Learning Systems

---

## ⚙️ Key Features

- Document ingestion (PDF / text)
- Text preprocessing and chunking
- Embedding generation for semantic search
- Vector similarity search using FAISS
- Context-aware prompt construction
- Response generation with LLaMA-based models
- Modular and extensible pipeline design

---

## 🏗️ System Architecture

**Pipeline Flow:**

1. Document Loading  
2. Text Chunking & Cleaning  
3. Embedding Generation  
4. Vector Storage (FAISS)  
5. Semantic Retrieval (Top-K)  
6. Context-Augmented Response Generation (LLaMA)

---

## 🧰 Tech Stack

- **Language:** Python  
- **LLM:** LLaMA / LLaMA-based models  
- **Libraries:** LangChain, Hugging Face Transformers  
- **Vector Database:** FAISS  
- **Embeddings:** Sentence Transformers  
- **Environment:** Jupyter Notebook  

---

## 📦 Installation

```bash
git clone https://github.com/dipdhru/medical_chatbot_rag.git
cd medical_chatbot_rag
pip install -r requirements.txt
