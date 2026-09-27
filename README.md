# EU AI Act Compliance RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system for answering regulatory compliance questions about the EU AI Act (Regulation (EU) 2024/1689). Built with an evaluation-first methodology: every architectural decision is backed by metrics from a curated golden dataset.

## Demo

```
Q: What obligations does Article 13 impose on providers of high-risk AI systems?

A: Article 13 requires that high-risk AI systems be designed and developed with
sufficient transparency to enable deployers to interpret the system's output and
use it appropriately (Article 13(1)). Providers must accompany high-risk AI
systems with instructions for use that include the identity and contact details
of the provider, the system's intended purpose, level of accuracy, and known
limitations (Article 13(3))...

⚠️ This is informational support for regulatory research only, not legal advice.
Consult qualified legal counsel for compliance decisions.
```

## Results

Final evaluation on held-out test split (n=13):

| Metric | Score | Target |
|---|---|---|
| Recall@1 | 1.000 | ≥ 0.60 |
| Recall@3 | 1.000 | ≥ 0.85 |
| MRR | 1.000 | ≥ 0.70 |
| Citation accuracy | 0.821 | ≥ 0.80 |
| Bad framing rate | 0.000 | ≤ 0.05 |
| Graceful failure rate | 1.000 | 1.00 |
| Answer relevance | 0.935 | ≥ 0.80 |
| Faithfulness | 0.965 | ≥ 0.80 |
| Factual consistency | 1.000 | ≥ 0.80 |

## Architecture

```
PDF → Structure-aware chunking → ChromaDB (dense) + BM25 (lexical)
                                        ↓
                              Hybrid retrieval (RRF)
                                        ↓
                            Cross-encoder reranker
                                        ↓
                        Claude Sonnet (cited generation)
                                        ↓
                         Citation verification + bad framing detection
```

**Key design decisions:**
- **Framework-agnostic** — direct Anthropic and OpenAI API calls only, no LangChain/LlamaIndex. Every component is transparent and independently testable.
- **Structural chunking** — one chunk per EU legal article/recital/annex, preserving legal context boundaries rather than splitting mid-article.
- **Hybrid retrieval** — dense vectors (semantic) fused with BM25 (keyword) via Reciprocal Rank Fusion. Outperforms dense-only on legal text with precise article references.
- **Cross-encoder reranker** — `cross-encoder/ms-marco-MiniLM-L-6-v2` applied after hybrid retrieval. Raised Recall@1 from 0.742 → 0.935 on dev split.
- **Three-layer evaluation** — retrieval metrics (Recall@k, MRR), automated response metrics (citation accuracy, graceful failure), and LLM-as-judge (faithfulness, factual consistency).

## Stack

| Component | Technology |
|---|---|
| LLM | Anthropic Claude Sonnet (`claude-sonnet-4-5`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (local) |
| Lexical search | `rank-bm25` (BM25Okapi) |
| Reranker | `sentence-transformers` cross-encoder |
| PDF parsing | PyMuPDF + pdfplumber fallback |
| API | FastAPI |
| Frontend | Streamlit |
| Package manager | `uv` + Python 3.14 |

## Experiments

Six controlled experiments, one variable changed at a time:

| Exp | Variable | Winner | Key finding |
|---|---|---|---|
| E1 | Hybrid vs dense-only retrieval | Hybrid | BM25 improves Recall@1 and MRR on legal text |
| E2 | Reranker on/off | Reranker on | Recall@1 +0.193, MRR +0.111 |
| E3 | top_k = 3 vs 5 | k=3 | k=5 hurt citation completeness and doubled cost |
| E4 | Prompt v1 vs v2 | v1 | Verbose graceful-failure rule regressed citation metrics |
| E5 | Structural vs fixed-size chunking | Inconclusive | Golden dataset chunk IDs are tied to structural boundaries |
| E6 | Sonnet vs Haiku | Sonnet | Haiku citation completeness dropped below target (0.667 vs 0.75) |

## Setup

**Prerequisites:** `uv`, an Anthropic API key, and an OpenAI API key.

```bash
# Clone and install
git clone <repo>
cd rag
uv sync

# Configure API keys
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY and OPENAI_API_KEY

# Ingest the EU AI Act PDF (place PDF in data/raw/ first)
uv run python -m src.ingestion.pipeline

# Run the Streamlit demo
uv run streamlit run app/main.py

# Run the FastAPI server
uv run uvicorn src.api.main:app --reload
```

## Evaluation

```bash
# Retrieval metrics (Recall@k, MRR, Precision@k)
uv run python -m eval.metrics.run_retrieval_eval --split dev

# Response metrics (citation accuracy, graceful failure rate)
uv run python -m eval.metrics.run_response_eval --split dev

# LLM-as-judge (faithfulness, factual consistency, answer relevance)
uv run python -m eval.judges.run_judge_eval --split dev

# Eval viewer (Streamlit)
uv run streamlit run eval/viewer/app.py
```

## Project Structure

```
src/
├── ingestion/    # PDF extraction, parsing, chunking, embedding, ChromaDB storage
├── retrieval/    # Dense search, BM25, hybrid RRF fusion, cross-encoder reranker
├── generation/   # Prompt templates, Anthropic API calls, citation verification
├── api/          # FastAPI endpoints, query orchestrator
└── config.py     # All tuneable values — one line change per experiment

eval/
├── golden_dataset/   # 44-query ground truth dataset (dev/test splits)
├── metrics/          # Retrieval and response evaluation runners
├── judges/           # LLM-as-judge prompts and runner (Claude Haiku)
└── viewer/           # Streamlit annotation and comparison tool

experiments/      # JSON results for every eval run
docs/             # Implementation plan, golden dataset methodology
```
