---
title: Medical RAG Chatbot
sdk: streamlit
emoji: 🏥
colorFrom: blue
colorTo: green
pinned: false
---

# Medical RAG Chatbot

[![CI](https://github.com/dipdhru/medical_chatbot_rag/actions/workflows/ci.yml/badge.svg)](https://github.com/dipdhru/medical_chatbot_rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

A retrieval-augmented chatbot that answers health questions from **16,407 MedQuAD Q&A pairs** published by NIH sources, cites where each answer came from, and refuses questions outside its knowledge base.

**[Live demo on Hugging Face Spaces](https://huggingface.co/spaces/dipdhrusingh/medical-chatbot)**

> For information only. This is a portfolio project, not medical advice.

## Results

Measured with [`scripts/evaluate_retrieval.py`](scripts/evaluate_retrieval.py) on a benchmark of 70 lay-language paraphrases and 40 out-of-scope questions. Half calibrates the rejection threshold; the figures below come from the unseen half (35 in-scope, 20 out-of-scope).

| Retrieval (top 3) | TF-IDF baseline | e5-base (this app) |
|---|---:|---:|
| Right condition found | 43% | **86%** (95% CI 71–94%) |
| Right condition and question type found | 9% | **49%** |

| Out-of-scope handling | Original threshold (0.50) | Calibrated (0.83) |
|---|---:|---:|
| Out-of-scope questions refused | 0 of 20 | **17 of 20** |
| In-scope questions answered | 35 of 35 | **35 of 35** |

**What this shows**

- Semantic search roughly doubles the hit rate over keywords, because people rarely use the clinical term. "Thick scaly red patches" needs to find psoriasis.
- The original threshold let everything through: even "fix a broken car engine" scored 0.79, because this model's cosine scores sit in a narrow, high band. A threshold only works if it is calibrated against real out-of-scope queries.
- Adding the `passage:` prefix recommended for e5 models made no measurable difference on this data (80% against 86%, well within the noise), so the existing embeddings were kept.

**Limits**

- The benchmark is small and was written for this project. It has not been reviewed by clinicians.
- The separation is narrow: the highest out-of-scope score (0.88) is above the lowest in-scope score (0.84), so some off-topic questions will still get through.
- These numbers cover retrieval and refusal only. The faithfulness of the generated answers has not been measured.

## How it works

```
Question → embed (e5-base) → FAISS top-k → similarity threshold → Llama 3.1 8B via Groq → answer + sources
```

| Component | Choice |
|---|---|
| Knowledge base | MedQuAD, 16,407 Q&A pairs from 9 NIH source collections |
| Embeddings | `intfloat/e5-base` (768 dimensions) |
| Search | FAISS `IndexFlatIP` on normalised vectors (cosine similarity) |
| Generation | Llama 3.1 8B via Groq, restricted to the retrieved context |
| Interface | Streamlit |

Without an API key, or if the LLM call fails, the app shows the best-matching source passage instead of failing.

## Run it

```bash
git clone https://github.com/dipdhru/medical_chatbot_rag.git
cd medical_chatbot_rag
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env   # optional, free at console.groq.com
export $(cat .env) && streamlit run app.py
```

The data and embeddings are stored with Git LFS, so run `git lfs install` before cloning.

## Test and evaluate

```bash
pip install pytest ruff
pytest                                 # no model download needed
ruff check .
python scripts/evaluate_retrieval.py   # rewrites results/retrieval_eval.json
```

## Project structure

```
├── app.py                  Streamlit interface
├── medrag/
│   ├── config.py           model names and defaults
│   ├── data.py             load MedQuAD, build retrieval text
│   ├── retrieval.py        FAISS search and out-of-scope threshold
│   ├── generation.py       grounded prompt, citations, fallback
│   └── metrics.py          hit rate, MRR, confidence intervals, calibration
├── evaluation/             benchmark questions
├── results/                evaluation output
├── scripts/                evaluation and embedding builds
├── tests/
└── RAG_llama.ipynb         original exploration notebook
```

The notebook is kept as a record of the first version. It uses the earlier 0.50 threshold and an 8-question evaluation, and its RAGAS step did not complete, so the figures above supersede it.

## Author

**Dipanshu Singh** · [LinkedIn](https://www.linkedin.com/in/dipanshu-s-b94384174/) · [GitHub](https://github.com/dipdhru)
