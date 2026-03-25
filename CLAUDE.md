# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production RAG pipeline for EU AI Act compliance Q&A. Target users are compliance officers and product managers querying EU AI Act regulatory text. The full spec is in `llm/PRD.md`.

## Package Manager & Commands

This project uses `uv` (not pip) with Python 3.14.

```bash
# Install dependencies
uv sync

# Add a dependency
uv add <package>

# All scripts must be run as modules from the project root (never as direct file paths)
# so that `src` and `eval` are importable.

# Ingest PDFs into ChromaDB
uv run python -m src.ingestion.pipeline

# Generate golden dataset candidates
uv run python -m eval.golden_dataset.generate

# Run retrieval evaluation
uv run python -m eval.metrics.run_retrieval_eval --split dev

# Run response evaluation
uv run python -m eval.metrics.run_response_eval --split dev

# Run LLM-judge evaluation
uv run python -m eval.metrics.run_judge_eval --split dev

# Run the FastAPI server
uv run uvicorn src.api.main:app --reload

# Run the Streamlit demo
uv run streamlit run app/main.py

# Run the eval viewer
uv run streamlit run eval/viewer/app.py

# Run tests
uv run pytest

# Run a single test
uv run pytest tests/unit/test_parser.py::test_parses_article_header -v
```

## Planned Architecture

The project follows a five-step production RAG framework: scoped MVP → golden dataset → baseline retrieval → evaluated response generation → controlled experimentation.

### Stack
- **LLM**: Anthropic Claude Sonnet (baseline) via `anthropic` SDK
- **Embeddings**: OpenAI `text-embedding-3-small` (baseline) via `openai` SDK
- **Vector store**: ChromaDB (local, zero-config)
- **Lexical search**: `rank_bm25` for hybrid retrieval alongside dense vectors
- **PDF processing**: PyMuPDF (`fitz`) or `pdfplumber`
- **Backend API**: FastAPI
- **Frontend / Eval viewer**: Streamlit
- **Testing**: pytest + custom eval harness

### Directory Structure (to be built)
```
src/
├── ingestion/      # PDF processing, chunking, embedding pipeline
├── retrieval/      # Dense vector search, BM25, hybrid retrieval fusion
├── generation/     # Prompt templates, LLM calls, citation logic
├── api/            # FastAPI endpoints
└── config.py       # Centralized configuration (models, thresholds, etc.)
eval/
├── golden_dataset/ # 40–50 query–answer–retrieval ground truth triples
├── metrics/        # Retrieval (Recall@k, MRR, Precision@k) and response eval code
├── judges/         # LLM-as-judge prompts for answer relevance, faithfulness, consistency
└── viewer/         # Streamlit annotation tool for manual error analysis
experiments/        # JSON/CSV logs for controlled experiments (one variable changed per run)
data/               # Source PDFs and processed chunks
app/                # Streamlit demo frontend
tests/              # pytest unit and integration tests
```

### RAG Pipeline Flow
1. **Ingestion**: PDFs → structure-aware chunking (one chunk per article/sub-article) → embed → store in ChromaDB with rich metadata
2. **Retrieval**: Hybrid search (dense vector + BM25) → optional metadata filtering by `section_type` → reranker (experiment E5)
3. **Generation**: Retrieved chunks injected into prompt → Claude Sonnet → response with inline citations (e.g., "Article 17(1)")
4. **Evaluation**: Automated metrics against golden dataset + LLM-judge for faithfulness and factual consistency

### Chunk Metadata Schema
Every ChromaDB chunk stores: `article_number`, `section_type` (obligation/definition/prohibition/procedural/recital), `title`, `chunk_index`, `parent_document`, `source_url`, `publication_date`, `ingestion_run_id`, `chunk_hash`.

### Retrieval Metrics Targets
- Recall@3 ≥ 0.85, Recall@5 ≥ 0.92, MRR ≥ 0.70, Precision@5 ≥ 0.60

### Response Metrics Targets
- Citation accuracy ≥ 0.80, Faithfulness ≥ 0.90, Factual consistency ≥ 0.90, Bad framing rate ≤ 0.05, Graceful failure rate = 100%

## Key Design Decisions

- **Framework-agnostic**: Direct API calls only — no LangChain/LlamaIndex. Keeps component behavior transparent and experiments easier to isolate.
- **One variable per experiment**: Experiment log tracks hypothesis, variable changed, and metric deltas. Never change multiple variables simultaneously.
- **Golden dataset is the most important artifact**: All pipeline improvements are measured against it; never merge changes that regress retrieval or response metrics.
- **Classic RAG first**: No agentic RAG until baseline failure modes are characterized.
- **MVP corpus**: EU AI Act main text + Annexes I–XIII + Recitals only. Commission guidance documents deferred until baseline metrics are proven.
