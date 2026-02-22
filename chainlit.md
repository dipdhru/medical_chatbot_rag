# Medical RAG Chatbot

Welcome! I'm a medical assistant powered by the **MedQuAD** knowledge base (16,407 Q&A pairs).

## How to use

- Ask any medical question in plain English
- Use the **Settings** panel (gear icon) to tune:
  - **Top-k documents** — how many sources to retrieve
  - **Similarity threshold** — minimum relevance score
  - **Show retrieved sources** — toggle source citations

## Example questions

- *What are the symptoms of type 2 diabetes?*
- *How is Marfan syndrome treated?*
- *What causes kidney stones?*

---

Powered by **intfloat/e5-base** embeddings · **Llama 3.1** via Groq · FAISS vector search
