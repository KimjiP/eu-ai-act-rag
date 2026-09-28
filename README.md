# EU AI Act Compliance RAG Pipeline

A retrieval-augmented question-answering system over the EU AI Act (Regulation (EU) 2024/1689). It answers with article-level citations, checks every citation against the text it retrieved, and says so when that text does not answer the question. Built evaluation-first: every change is measured on a golden dataset, one variable at a time.

**Corpus:** the Regulation as published in the Official Journal on 12 July 2024. Later amendments, including the Digital Omnibus on AI (Regulation (EU) 2026/1744, in force since 27 July 2026), are not included, and the app shows this next to every answer.

## Demo

```
Q: What does 'AI literacy' mean in the AI Act?

A: 'AI literacy' means skills, knowledge and understanding that allow providers,
deployers and affected persons [...] to make an informed deployment of AI systems,
as well as to gain awareness about the opportunities and risks of AI and possible
harm it can cause (Article 3(56)). [...] Providers and deployers must take
measures to ensure a sufficient level of AI literacy of their staff (Article 4).

Citation check: Article 3(56) ✓, Recital 20 ✓, Article 4 ✓
```

```
Q: Which authority supervises the AI Act in Norway?

Declined. The provided regulatory text does not contain sufficient information to
answer this question. [...] You would need Norway's incorporation of the AI Act
through the EEA Agreement and the national legislation designating authorities.
```

## September 2026 audit

Before showing this project to anyone, I audited it. The pipeline design held up; the data under it and the evaluation around it did not.

**What was wrong**

1. **The parser mislabelled a third of the corpus.** It started a new article on any line beginning with "Article N", including cross-references that wrap onto a new line in the PDF, and read Article 3's numbered definitions as recitals. All 180 recitals ended up in two chunks of 98,000 and 147,000 characters (the larger labelled "Article 39"), at least 33 chunks carried the label of an article they only mention, and definitions such as Article 3(56) were labelled "Recital 56". Asked what "AI literacy" means, the system quoted the right text and cited it as Recital 56, which is about AI in education.
2. **The citation check compared strings by prefix**, so a citation to Article 13 passed when only Article 1 was retrieved, and Annex III passed on Annex I.
3. **The evaluation could not catch either.** Its answer key was built by letting an LLM pick the relevant chunks from the retriever's own top 5, seeing only the first 600 characters of each, so retrieval was graded against the retriever's earlier choices. The test split had no unanswerable questions, and a declined answer was never detected, so the reported graceful-failure score of 1.000 was never measured.

**What changed**

- **Parser** ([src/ingestion/parser.py](src/ingestion/parser.py)): removes page furniture and footnotes, accepts a header only as a whole line in its region of the document and in sequence, and splits Article 3 into one unit per definition. The corpus is now 374 units: the preamble, 180 recitals, 112 articles plus Article 3's 68 definitions, and 13 annexes, all uniquely labelled. Tests reproduce each failure and check the real PDF.
- **Citations** are compared as provisions (Article 17(1) matches Article 17; Article 13 never matches Article 1), references to other acts are skipped, and declines are detected.
- **Retrieval**: the reranker scores 20 candidates instead of reordering the final 3, and recitals rank below the operative text (E9 below).
- **Evaluation**: ground truth comes from the text, never from chunk labels ([eval/provisions.py](eval/provisions.py)). Golden set v2 has 44 answerable questions labelled by the provisions that answer them, and 10 the corpus cannot answer. One runner computes every metric on the same answer, and a completeness judge checks answers against the Act itself, not only against what was retrieved.

## Results

Each column changes one thing:

- **v1-original**: the system as built in March 2026
- **v2-parser-only**: corrected parse (corpus v2)
- **v2-rerank20**: reranker scores 20 candidates instead of reordering the final 3
- **v2-final**: recitals ranked below the operative text

All runs use the same model (Claude Sonnet 4.5, prompt v1, 3 passages per answer) and the same evaluation. Judges are Claude Haiku 4.5.

### Test split (nothing was tuned on it)

13 answerable and 4 unanswerable questions.

| Metric | v1-original | v2-parser-only | v2-rerank20 | v2-final |
|---|---|---|---|---|
| Recall@1 | 0.85 | 0.69 | 0.61 | 0.85 |
| Recall@3 | 1.00 | 0.77 | 0.85 | 1.00 |
| Recall@5 | 1.00 | 0.92 | 1.00 | 1.00 |
| MRR | 0.91 | 0.77 | 0.75 | 0.91 |
| Citations: cited text was retrieved | 0.81 | 0.84 | 0.83 | 0.85 |
| Citations: label of a passage whose text is another provision | 0.03 | 0.00 | 0.00 | 0.00 |
| Citations: cross-reference named in retrieved text | 0.16 | 0.16 | 0.17 | 0.15 |
| Citations: unsupported | 0.00 | 0.00 | 0.00 | 0.00 |
| Unanswerable questions declined | 1.00 | 1.00 | 1.00 | 1.00 |
| Answerable questions declined | 0.00 | 0.08 | 0.00 | 0.00 |
| Completeness vs the Act (judge) | 0.92 | not run | 0.84 | 0.96 |
| Faithfulness to retrieved text (judge) | 0.93 | not run | 0.99 | 1.00 |
| Factual consistency (judge) | 0.96 | not run | 1.00 | 1.00 |
| Answer relevance (judge) | 0.92 | not run | 0.94 | 0.95 |
| Avg characters sent to the LLM | 7,376 | 7,485 | 7,212 | 9,159 |
| Avg cost per question (USD) | $0.0113 | $0.0110 | $0.0108 | $0.0124 |

### Dev split (the recital penalty was tuned on it)

31 answerable and 6 unanswerable questions.

| Metric | v1-original | v2-parser-only | v2-rerank20 | v2-final |
|---|---|---|---|---|
| Recall@1 | 0.68 | 0.52 | 0.48 | 0.68 |
| Recall@3 | 0.94 | 0.74 | 0.68 | 0.81 |
| Recall@5 | 0.94 | 0.81 | 0.74 | 0.84 |
| MRR | 0.79 | 0.63 | 0.58 | 0.74 |
| Citations: cited text was retrieved | 0.83 | 0.81 | 0.85 | 0.81 |
| Citations: label of a passage whose text is another provision | 0.04 | 0.00 | 0.00 | 0.00 |
| Citations: cross-reference named in retrieved text | 0.13 | 0.17 | 0.14 | 0.18 |
| Citations: unsupported | 0.00 | 0.01 | 0.00 | 0.00 |
| Unanswerable questions declined | 1.00 | 1.00 | 1.00 | 1.00 |
| Answerable questions declined | 0.03 | 0.03 | 0.07 | 0.07 |
| Completeness vs the Act (judge) | 0.73 | not run | 0.69 | 0.81 |
| Faithfulness to retrieved text (judge) | 0.96 | not run | 0.98 | 0.98 |
| Factual consistency (judge) | 0.95 | not run | 1.00 | 0.99 |
| Answer relevance (judge) | 0.91 | not run | 0.90 | 0.88 |
| Avg characters sent to the LLM | 11,791 | 6,673 | 6,612 | 8,093 |
| Avg cost per question (USD) | $0.0147 | $0.0111 | $0.0111 | $0.0122 |

For comparison, the March 2026 evaluation reported Recall@1, Recall@3 and MRR of 1.000, citation accuracy 0.821 and graceful failure 1.000 on 13 test questions.

**What the numbers say**

- **Citations now point at the right provision.** The original system cited the label of a passage whose text was a different provision in 3 to 4% of its citations (11 citations, such as "Recital 56" for the definition of AI literacy). The corrected system: none.
- **Answers are more complete and more faithful.** Completeness against the Act went from 0.92 to 0.96 on test and from 0.73 to 0.81 on dev. Faithfulness to the retrieved text went from 0.93 to 1.00 on test and from 0.96 to 0.98 on dev.
- **Retrieval is equal on test and lower on dev at 3 and 5 passages.** Recall@1 is unchanged on both splits (0.85 and 0.68). On dev, Recall@3 is 0.81 against 0.94. Two things favour the original here: the questions were generated from its chunks, and recall counts a passage as a hit when it contains a gold provision's text anywhere, which rewards larger chunks (the original sent 11,800 characters per answer on dev, against 8,100).
- **Fixing the parser first made retrieval worse.** As separate chunks, recitals outranked the articles that carry the obligations (v2-parser-only, v2-rerank20). Ranking them below the operative text (v2-final) recovered it.
- **Declines.** Every unanswerable question in both splits is declined. The final system declines 2 of 31 answerable dev questions (one retrieval miss, and one borderline question on whether Article 4 sets a number of training hours, where it declines instead of answering "no"), against 1 for the original.
- **Cost** is about $0.012 per question: 17% less than the original on dev, 10% more on test.

The splits are small: one question moves a recall score by 0.08 on test and 0.03 on dev.

## Architecture

```
PDF → OJ-aware structural parser → ChromaDB (dense) + BM25 (lexical)
                                            ↓
                                  Hybrid retrieval (RRF), 20 candidates
                                            ↓
                          Cross-encoder reranker, recitals ranked below operative text
                                            ↓
                              Claude Sonnet (cited generation), top 3 passages
                                            ↓
                     Citation check against the retrieved passages, decline detection
```

**Key design decisions:**
- **Framework-agnostic**: direct Anthropic and OpenAI API calls only, no LangChain or LlamaIndex. Every component is transparent and independently testable.
- **Structural chunking**: one chunk per article, recital and annex, and one per definition in Article 3, so answers can cite the exact provision.
- **Hybrid retrieval**: dense vectors fused with BM25 via Reciprocal Rank Fusion; BM25 matters for precise article references.
- **Cross-encoder reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` over 20 hybrid candidates.
- **Evaluation that does not trust the system**: gold labels are provisions, and passages are matched to provisions by their text.

## Evaluation

```bash
# Run the evaluation on one split: retrieval, answers, citations, declines, judges
uv run python -m eval.run_eval --split test --tag my-change

# The same evaluation on the original system
RAG_CORPUS_VERSION=v1 RAG_RERANK_CANDIDATES=0 RAG_RECITAL_PENALTY=0 \
  uv run python -m eval.run_eval --split test --tag v1-original

# Compare saved runs as a markdown table
uv run python -m eval.compare_runs --split test v1-original my-change

# Browse runs question by question (Streamlit)
uv run streamlit run eval/viewer/app.py
```

What is measured:

| Metric | How |
|---|---|
| Recall@k, MRR | Does a top-k passage contain a gold provision? Passages are mapped to provisions by their text, so wrong labels cannot inflate the score. |
| Citation status | For each cited provision: its text was retrieved; or the retrieved text names it (a cross-reference, e.g. Article 9 refers to "the post-market monitoring system referred to in Article 72"); or unsupported. |
| Declines | Share of unanswerable questions declined (should be 1) and of answerable ones declined (should be 0). A decline is an answer that opens with the decline sentence the system prompt specifies; an answer that says later that part of the question is not covered counts as answered. |
| Completeness (judge) | Share of the answer key's facts the answer states. The facts come from the Act, so this catches answers built on the wrong passages. |
| Faithfulness, consistency, relevance (judges) | The answer against the retrieved passages and the question. |

The golden dataset and how v2 was built: [docs/golden_dataset_methodology.md](docs/golden_dataset_methodology.md) and [eval/golden_dataset/build_v2.py](eval/golden_dataset/build_v2.py).

## Experiments

E1 to E6 were run in March 2026 with the v1 evaluation, whose labels came from the retriever's own results. Treat them as directional.

| Exp | Variable | Winner | Key finding |
|---|---|---|---|
| E1 | Hybrid vs dense-only retrieval | Hybrid | BM25 improves Recall@1 and MRR on legal text |
| E2 | Reranker on/off | Reranker on | Recall@1 +0.193, MRR +0.111 |
| E3 | top_k = 3 vs 5 | k=3 | k=5 hurt citation completeness and doubled cost |
| E4 | Prompt v1 vs v2 | v1 | Verbose graceful-failure rule regressed citation metrics |
| E5 | Structural vs fixed-size chunking | Inconclusive | Golden dataset chunk IDs are tied to structural boundaries |
| E6 | Sonnet vs Haiku | Sonnet | Haiku citation completeness dropped below target (0.667 vs 0.75) |
| E7 | Corrected parser (September 2026) | v2 | No more wrong-label citations, but recitals, now separate chunks, outranked the articles: dev Recall@3 0.94 → 0.74 |
| E8 | Rerank 20 candidates vs the final 3 | 20 | Mixed on its own (Recall@3: test 0.77 → 0.85, dev 0.74 → 0.68); kept because E9 can only promote an article that is in the candidate pool |
| E9 | Recital penalty 0 to 6, or exclude | 3.0 | Tuned on dev: Recall@1 0.48 → 0.68, Recall@3 0.68 → 0.81, recitals in the top 3 from 1.6 to 0.4 per question. On test: Recall@1 0.61 → 0.85 |

The March 2026 result files are in [experiments/archive-2026-03/](experiments/archive-2026-03/).

## Stack

| Component | Technology |
|---|---|
| LLM | Anthropic Claude Sonnet 4.5 (`claude-sonnet-4-5`) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (local) |
| Lexical search | `rank-bm25` (BM25Okapi) |
| Reranker | `sentence-transformers` cross-encoder |
| PDF parsing | PyMuPDF + pdfplumber fallback |
| API | FastAPI |
| Frontend | Streamlit |
| Package manager | `uv` + Python 3.14 |

## Setup

**Prerequisites:** `uv`, an Anthropic API key, and an OpenAI API key.

```bash
git clone https://github.com/KimjiP/eu-ai-act-rag.git
cd eu-ai-act-rag
uv sync

# Configure API keys
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY and OPENAI_API_KEY

# Download the Regulation from EUR-Lex into data/raw/
curl -L -o data/raw/OJ_L_202401689_EN_TXT.pdf \
  "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689"

# Build the index (about $0.003 of embeddings)
uv run python -m src.ingestion.pipeline

# Run the Streamlit demo
uv run streamlit run app/main.py

# Run the FastAPI server
uv run uvicorn src.api.main:app --reload

# Unit tests (no API calls)
uv run pytest
```

## Project Structure

```
src/
├── ingestion/    # PDF extraction, OJ-aware structural parser, chunking, embedding, ChromaDB
├── retrieval/    # Dense search, BM25, hybrid RRF fusion, reranker, recital ranking
├── generation/   # Prompt templates, Anthropic API calls, citation check, decline detection
├── api/          # FastAPI endpoints, query orchestrator
└── config.py     # All tuneable values; one line change per experiment

eval/
├── golden_dataset/   # Golden set v2 (54 questions) and its build script; v1 kept for reference
├── provisions.py     # Ground truth: which provisions a passage's text contains
├── run_eval.py       # One runner: retrieval, answers, citations, declines, judges
├── compare_runs.py   # Markdown table of saved runs
├── metrics/          # Retrieval and response metric functions
├── judges/           # LLM-as-judge prompts (Claude Haiku)
└── viewer/           # Streamlit viewer for runs, per question

experiments/      # Saved evaluation runs; March 2026 runs in archive-2026-03/
docs/             # Implementation plan, golden dataset methodology
```

## Limitations and next steps

- **Corpus version**: the 2024 text only. The 2026 amendments are not ingested, so answers about application dates of high-risk obligations reflect the original Article 113.
- **English only**: the reranker (`ms-marco-MiniLM`) is trained on English. Other languages need a multilingual reranker, re-evaluated.
- **Long articles**: the reranker reads the first 512 tokens of a passage, so the end of a long article (Article 5 is 11,000 characters) cannot help it rank. Paragraph-level chunks for long articles are the next experiment.
- **Golden set size**: 54 questions (17 in test). Labels were reviewed by Claude against the full text, not yet by a person; small differences between runs are within noise.
- **Judges are LLMs** (Claude Haiku 4.5) and can be wrong; the deterministic metrics (recall, citation status, declines) do not depend on them.
