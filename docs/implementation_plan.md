# Implementation Plan: EU AI Act Compliance RAG Pipeline

## Phase 0 — Repository Scaffolding (Day 0)

**0.1 Add all dependencies** to `pyproject.toml` and run `uv sync`:
- `anthropic`, `chromadb`, `rank-bm25`, `pymupdf`, `fastapi`, `uvicorn[standard]`, `streamlit`, `pytest`, `pytest-asyncio`, `sentence-transformers`, `httpx`, `python-dotenv`, `ruff`, `pydantic`

**0.2 Create directory skeleton:**
```
src/{ingestion,retrieval,generation,api}/
eval/{golden_dataset,metrics,judges,viewer}/
experiments/ data/{raw,processed}/ app/ tests/{unit,integration}/ docs/
```

**0.3 Create `src/config.py`** — single source of truth for every tuneable value: model names, `TOP_K_RETRIEVAL`, `CONFIDENCE_THRESHOLD`, `CHROMA_COLLECTION_NAME`, all directory paths. Every experiment changes exactly one entry here.

---

## Phase 1 — Foundation & Data (Week 1)

**1.1** Download EU AI Act PDFs to `data/raw/` (main text, Annexes I–XIII, Recitals).

**1.2 `src/ingestion/extractor.py`** — `extract_pdf(path) -> list[RawPage]` using PyMuPDF, pdfplumber as fallback.

**1.3 `src/ingestion/parser.py`** *(highest-risk file)* — `parse_document(pages) -> list[StructuralUnit]`. Uses regex to detect EU legal format: `Article \d+`, recital numbering `(\d+)`, annex headers `ANNEX [IVX]+`. Assigns `SectionType` enum (OBLIGATION, DEFINITION, PROHIBITION, PROCEDURAL, RECITAL, ANNEX).

**1.4 `src/ingestion/chunker.py`** — defines the `Chunk` dataclass (the interface between ingestion and retrieval) plus two strategies:
- `chunk_structural()` — one chunk per structural unit (baseline)
- `chunk_fixed_size(token_limit=500, overlap=50)` — used in Experiment E1

**1.5** `src/ingestion/embedder.py` → batch OpenAI embedding calls + cost logging. `src/ingestion/store.py` → upsert to ChromaDB using `chunk_hash` as ID (idempotent). `src/ingestion/pipeline.py` → orchestrates everything; run with `uv run python -m src.ingestion.pipeline`.

**1.6 Tests:** `tests/unit/test_parser.py`, `test_chunker.py`, `test_embedder.py` (mocked), `tests/integration/test_ingestion_pipeline.py`.

**Exit criterion:** ChromaDB has 200+ chunks, `uv run pytest tests/` is green.

---

## Phase 2 — Golden Dataset & Baseline Retrieval (Week 2)

**2.1 `eval/golden_dataset/generate.py`** — feeds chunks to Claude Sonnet to generate candidate queries. Output to `candidates_raw.json`. Manually curate into `golden_dataset.json` with schema: `query_id`, `query`, `type`, `difficulty`, `persona`, `split`, `relevant_chunk_ids`, `expected_answer_elements`, `is_unanswerable`. **This file is never auto-modified after review.**

**2.2 `src/retrieval/dense.py`** — `search_dense(query, top_k, filters) -> list[RetrievalResult]`. `RetrievalResult` is the interface between retrieval and generation (chunk_id, text, score, rank, article_number, section_type).

**2.3 `src/retrieval/bm25.py`** — `BM25Index` class, built from `data/processed/chunks_v1.json` at startup.

**2.4 `src/retrieval/hybrid.py`** — `search_hybrid(query, top_k, alpha=0.5)` using Reciprocal Rank Fusion (RRF, k=60) to merge dense + BM25 ranked lists.

**2.5 `eval/metrics/retrieval.py`** — `compute_recall_at_k()`, `compute_mrr()`, `compute_precision_at_k()`, `evaluate_retrieval(dataset_path, split) -> RetrievalMetrics`.

Run `uv run python eval/metrics/run_retrieval_eval.py --split dev`. Record results as **E0 baseline** in `experiments/E0_baseline.json`.

**Exit criterion:** Retrieval eval script runs to completion. E0 baseline metrics saved.

---

## Phase 3 — Response System & Error Analysis (Week 3)

**3.1 `src/generation/prompts.py`** — `SYSTEM_PROMPT` (cite every claim with article/paragraph, never hallucinate, graceful decline, legal disclaimer) + `build_user_prompt(query, chunks)`.

**3.2 `src/generation/llm.py`** — `generate_response() -> LLMResponse` (answer, token counts, latency, cost_usd).

**3.3 `src/generation/citations.py`** — `extract_citations(answer)` via regex, `verify_citations(citations, chunks) -> CitationVerification`.

**3.4 `src/api/query.py`** — `answer_query(query) -> QueryResponse`. Orchestrates: search → confidence check → generate → verify. Graceful failure returns `answered=False` without calling the LLM.

**3.5 `src/api/main.py`** — FastAPI with `POST /query`, `GET /health`, `GET /chunks/{chunk_id}`. Request/response logging middleware writes to `experiments/query_log.jsonl`.

**3.6 `eval/metrics/response.py`** — `compute_citation_accuracy()`, `compute_citation_completeness()`, `detect_bad_framing()` (regex for banned phrases), `check_graceful_failure()`.

**3.7 `eval/judges/`** — LLM-as-judge prompts (answer relevance, faithfulness, factual consistency) using Claude Haiku to keep costs low. `llm_judge.py` runs them and parses JSON scores.

**3.8 `eval/viewer/app.py`** — Streamlit eval viewer: per-query display of retrieved chunks + generated answer + metrics. Side-by-side mode for before/after experiment comparison. Annotation panel saves failure classes to `error_analysis_template.json`.

**Exit criterion:** 50+ responses manually reviewed and annotated. Failure class distribution drives experiment priority for Phase 4.

---

## Phase 4 — Experimentation (Week 4)

Each experiment changes **exactly one variable**, is logged to `experiments/experiment_log.json`, and is adopted or rejected before the next begins.

| Exp | Variable | Files Changed |
|-----|----------|---------------|
| E0 | Baseline | None — first run |
| E1 | Structure-aware vs. fixed-size chunking | `src/ingestion/pipeline.py` (flag), `src/config.py` (collection name) |
| E2 | Add BM25 hybrid | `src/config.py` (`RETRIEVAL_STRATEGY`), `src/retrieval/__init__.py` |
| E3 | Metadata filtering by section_type | `src/api/query.py`, `eval/metrics/run_retrieval_eval.py` |
| E4 | Prompt engineering | `src/generation/prompts.py`, `src/config.py` (`PROMPT_VERSION`) |
| E5 | Cross-encoder reranker | `src/retrieval/reranker.py` (new), `src/retrieval/__init__.py`, `src/config.py` |
| E6 | Embedding model upgrade | `src/config.py` (`EMBEDDING_MODEL`) only |

**Exit criterion:** 3+ experiments logged, retrieval targets met on dev split (Recall@3 ≥ 0.85, MRR ≥ 0.70), response targets met (citation accuracy ≥ 0.80, faithfulness ≥ 0.90, bad framing ≤ 0.05, graceful failure = 100%).

---

## Phase 5 — Polish & Delivery (Week 5)

**5.1 `app/main.py`** — Streamlit demo with query input, answer display, citation highlights, expandable source chunks, disclaimer banner.

**5.2 `Dockerfile`** — multi-stage build, default cmd runs FastAPI. `.dockerignore` excludes `data/raw/`, `.venv/`.

**5.3 Docs** — `README.md` with architecture diagram + eval results table, `docs/experiment_report.md` (narrative over `experiment_log.json`), `docs/architecture.md` (full data flow).

**5.4 Final test pass** — run full suite against the **test split** (never touched during development):
```bash
uv run pytest tests/ -v
uv run python eval/metrics/run_retrieval_eval.py --split test
uv run python eval/metrics/run_response_eval.py --split test
uv run python eval/metrics/run_judge_eval.py --split test
```
Test-split numbers go in the README and portfolio writeup.

---

## Key Data Flow

```
PDF → extractor → parser → chunker → embedder → ChromaDB
                                    ↘ BM25Index (from JSON)
Query → hybrid fusion (RRF) → [reranker] → prompt builder → Claude Sonnet → citation extractor → QueryResponse
```

## Key Interfaces Between Components

| From | To | Interface |
|------|----|-----------|
| `src/ingestion/chunker.py` | `src/retrieval/*.py` | `Chunk` dataclass (via `data/processed/chunks_v1.json` + ChromaDB) |
| `src/retrieval/__init__.py` | `src/generation/llm.py` | `list[RetrievalResult]` |
| `src/generation/llm.py` | `src/api/query.py` | `LLMResponse` |
| `src/api/query.py` | `src/api/main.py` and `app/main.py` | `QueryResponse` |
| `eval/golden_dataset/golden_dataset.json` | `eval/metrics/*.py` | JSON schema (query_id, relevant_chunk_ids, is_unanswerable) |
| `eval/metrics/*.py` | `experiments/experiment_log.json` | `RetrievalMetrics` + `ResponseMetrics` dicts |

## Three Most Critical Files

1. **`src/config.py`** — experiment control; every other module reads from here
2. **`src/ingestion/parser.py`** — EU legal structure parsing; bugs here corrupt all chunk metadata and invalidate all downstream metrics
3. **`eval/golden_dataset/golden_dataset.json`** — all metrics are only as good as this file; never auto-modify after manual review
