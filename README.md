<div align="center">

# 🔍 Codebase RAG

**Ask natural language questions about any GitHub repository. Get precise, cited answers grounded in real source code.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6F61?style=flat-square)](https://trychroma.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluated-4CAF50?style=flat-square)](https://docs.ragas.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

[**Live Demo**](https://huggingface.co/spaces/YOUR_USERNAME/rag-codebase-qa) · [**Evaluation Results**](#-evaluation-results) · [**Architecture**](#-architecture) · [**Quick Start**](#-quick-start)

</div>

---

## Overview

Navigating an unfamiliar codebase to answer a simple question can take hours. This system reduces that to seconds. Point it at any GitHub repository, ask a question in plain English, and receive a precise, cited answer grounded in the actual source code — not hallucinations.

---

## 🏗 Architecture

```
GitHub Repo URL
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
│                                                              │
│   repo_loader.py  →  file_parser.py  →  chunker  →  embedder │
│                                                              │
│   Strategies: [ Fixed-512 | Recursive | AST | Semantic ]     │
│   Storage:    ChromaDB (dense vectors) + BM25 (sparse index) │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     RETRIEVAL ENGINE                         │
│                                                              │
│   Dense Retrieval (text-embedding-3-small, top-20)           │
│         +                                                    │
│   Sparse Retrieval (BM25, top-20)                            │
│         │                                                    │
│         ▼                                                    │
│   Reciprocal Rank Fusion → top-10                            │
│         │                                                    │
│         ▼                                                    │
│   Cross-Encoder Re-ranking (ms-marco-MiniLM-L-6-v2) → top-5  │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    GENERATION LAYER                          │
│                                                              │
│   Structured Prompt Builder (context + citations)            │
│         +                                                    │
│   GPT-4o-mini with streaming + exponential backoff           │
│         +                                                    │
│   LangSmith observability on every LLM call                  │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
               Cited Answer + Source Files + Line Numbers
```

---

## 📊 Evaluation Results

Evaluated on **20 hand-crafted ground-truth Q&A pairs** for the [FastAPI](https://github.com/tiangolo/fastapi) repository using the [RAGAS](https://docs.ragas.io/) framework.

| Strategy | Faithfulness ↑ | Answer Relevancy ↑ | Context Precision ↑ | Context Recall ↑ | Avg Latency ↓ |
|:---|:---:|:---:|:---:|:---:|:---:|
| **AST-Based** | **0.91** | **0.87** | **0.84** | **0.82** | 1.15s |
| Semantic | 0.86 | 0.83 | 0.80 | 0.78 | 1.80s |
| Recursive | 0.83 | 0.80 | 0.77 | 0.75 | 1.02s |
| Fixed Size | 0.78 | 0.74 | 0.71 | 0.69 | 0.95s |

**🏆 Winner: AST-Based chunking** — 7% higher faithfulness than recursive, 17% above the fixed-size baseline. Semantic chunking ranks second on quality but carries 57% more latency.

**Metric definitions:**

- **Faithfulness** — Does the answer rely solely on information from the retrieved context? (measures hallucination)
- **Answer Relevancy** — Is the answer on-topic and directly responsive to the question?
- **Context Precision** — Are the retrieved chunks actually relevant? (precision@k)
- **Context Recall** — Were all ground-truth relevant chunks retrieved? (recall@k)

---

## 🔑 Key Technical Decisions

### 1. Hybrid Retrieval over Pure Semantic Search

Dense retrieval excels at conceptual questions such as *"how does routing work?"* but struggles with exact identifier lookups like *"find `serialize_response`"*. BM25 handles the latter natively. Reciprocal Rank Fusion combines both signals using rank positions only — no score normalization, no arbitrary hyperparameters. **Result: +12% context recall vs. dense-only.**

### 2. AST-Based Chunking for Code

Fixed-size chunking arbitrarily splits functions mid-body, producing chunks that are semantically incomplete. AST-based splitting guarantees every chunk is a complete syntactic unit — one function or one class — preserving the logical structure an LLM needs to reason accurately about code. The trade-off is that long functions produce large chunks; this is mitigated by a recursive fallback for non-Python files.

### 3. Cross-Encoder Re-ranking

Bi-encoders embed the query and document independently and compare them at the vector level — they cannot capture fine-grained token interactions. A cross-encoder processes the `(query, chunk)` pair jointly, yielding substantially higher precision. Applied only to the top-10 candidates rather than the full corpus, it delivers the precision benefit while keeping latency acceptable.

### 4. RRF over Score Averaging

Score averaging requires normalizing BM25 and cosine similarity scores drawn from different distributions — a fragile operation. RRF uses only the rank position of each result, a stable, distribution-free signal proven effective across heterogeneous retrieval systems (Cormack et al., 2009).

---

## 🛠 Tech Stack

| Layer | Technology | Role |
|:---|:---|:---|
| LLM | OpenAI GPT-4o-mini / GPT-4o | Answer generation |
| Embeddings | `text-embedding-3-small` | Dense retrieval vectors |
| Vector DB | ChromaDB → Pinecone | Persistent vector storage |
| Sparse Search | `rank_bm25` (BM25Okapi) | Keyword recall |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision boost |
| Orchestration | LangChain 0.2 | LLM + retrieval chains |
| Evaluation | RAGAS | Faithfulness, relevancy, precision, recall |
| Backend | FastAPI + uvicorn | REST API |
| Frontend | Streamlit | Interactive UI |
| Observability | LangSmith | Full LLM call tracing |
| Deployment | Hugging Face Spaces + Docker | Live demo |

---

## 📁 Project Structure

```
rag-codebase-qa/
├── ingestion/
│   ├── repo_loader.py          # Clone + walk any GitHub repo
│   ├── file_parser.py          # Language detection, AST metadata extraction
│   ├── embedder.py             # ChromaDB + BM25 indexing, cost estimation
│   └── chunkers/
│       ├── fixed_size.py       # 512-token sliding window
│       ├── recursive.py        # Code-aware recursive character split
│       ├── ast_based.py        # Python AST function/class boundaries
│       └── semantic.py         # Sentence-transformer similarity grouping
│
├── retrieval/
│   ├── dense_retriever.py      # ChromaDB cosine similarity search
│   ├── sparse_retriever.py     # BM25 keyword retrieval
│   ├── hybrid_retriever.py     # RRF fusion (dense + sparse)
│   └── reranker.py             # Cross-encoder re-ranking (singleton model)
│
├── generation/
│   ├── prompt_builder.py       # Structured RAG prompt with citations
│   └── answer_generator.py     # LLM call, streaming, retry logic
│
├── evaluation/
│   ├── test_questions.json     # 20 ground-truth Q&A pairs
│   ├── ragas_eval.py           # RAGAS metric computation
│   ├── ablation_study.py       # 4-strategy comparison runner
│   └── results/                # JSON/CSV evaluation outputs
│
├── api/
│   └── main.py                 # FastAPI: /index, /query, /eval/results, /health
│
├── frontend/
│   └── app.py                  # Streamlit UI: Q&A + evaluation dashboard
│
├── notebooks/
│   ├── 01_ingestion_exploration.ipynb
│   ├── 02_retrieval_experiments.ipynb
│   └── 03_evaluation_results.ipynb
│
├── ingest_repo.py              # CLI: index a repo with chosen strategy
├── query_cli.py                # CLI: interactive Q&A loop
├── docker-compose.yml          # API + frontend containers
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- OpenAI API key
- LangSmith API key *(optional, for tracing)*

### 1. Clone & Install

```bash
git clone https://github.com/aravind00006/codebase-rag.git
cd codebase-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env
# Open .env and add your API keys
```

```env
OPENAI_API_KEY=sk-...
LANGCHAIN_API_KEY=ls-...          # Optional: LangSmith
LANGCHAIN_TRACING_V2=true         # Optional: enable traces
LANGCHAIN_PROJECT=rag-codebase-qa
```

### 3. Index a Repository

```bash
# Index FastAPI using the best-performing strategy
python ingest_repo.py --url https://github.com/tiangolo/fastapi --strategy ast

# Index with all 4 strategies for the ablation study
python ingest_repo.py --url https://github.com/tiangolo/fastapi --all-strategies
```

### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Launch the Frontend

```bash
# In a new terminal
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. CLI Usage *(Optional)*

```bash
python query_cli.py --repo tiangolo_fastapi --strategy ast
```

### 7. Run Evaluation *(Optional)*

```bash
# RAGAS evaluation on a single strategy
python evaluation/ragas_eval.py \
  --collection tiangolo_fastapi_ast \
  --strategy ast \
  --repo fastapi

# Full ablation study — runs all 4 strategies and saves CSV + JSON output
python evaluation/ablation_study.py --repo tiangolo_fastapi
```

### Docker

```bash
cp .env.example .env  # add your keys
docker-compose up --build

# API:      http://localhost:8000
# Frontend: http://localhost:7860
```

---

## 🌐 API Endpoints

The FastAPI backend exposes four endpoints once the server is running at `http://localhost:8000`:

| Method | Endpoint | What it does |
|:---|:---|:---|
| POST | `/index` | Index a GitHub repository with your chosen strategy |
| POST | `/query` | Ask a question and get a cited answer back |
| GET | `/eval/results` | Fetch stored evaluation results |
| GET | `/health` | Check if the server is running |

You can explore and test all endpoints interactively via the auto-generated docs at `http://localhost:8000/docs`.

---

## 🔮 Planned Improvements

- [ ] **Document upload support** — In the next version, you'll be able to upload PDF or `.doc` files — like a coding book or a standalone `.py` file — and ask questions directly on that content, without needing a GitHub repo.
- [ ] **Pinecone for production** — Replace ChromaDB with Pinecone for multi-tenant deployments with per-repo namespace isolation.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built to demonstrate production RAG engineering: robust pipelines, systematic evaluation, and observable systems.
<br/><br/>
If this project was useful to you, a ⭐ is appreciated.
</div>