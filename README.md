# EU AI Act Q&A (RAG)

A question-answering system over the EU AI Act (Regulation (EU) 2024/1689). It answers with the article, recital or annex behind each statement, checks every citation against the passages it retrieved, and says so when the Act does not answer the question.

Companion code for the Signal Syntax post [Building and Evaluating a RAG System in Five Steps: An EU AI Act Example](https://signalsyntax.no/blog/eu-ai-act-rag/).

```
Q: What does 'AI literacy' mean in the AI Act?

A: 'AI literacy' means skills, knowledge and understanding that allow providers,
deployers and affected persons [...] (Article 3(56)). [...] Providers and deployers
must take measures to ensure a sufficient level of AI literacy of their staff (Article 4).

Citation check: Article 3(56) ✓, Recital 20 ✓, Article 4 ✓
```

## How it works

```
PDF → parser, one piece per provision → embeddings (ChromaDB) + BM25
    → merged (RRF) → cross-encoder reranker over 20 candidates, recitals below articles
    → Claude Sonnet 4.5 answers from the top 3 → citation check, decline detection
```

## Results

Baseline against the final system, on the same 54 questions (17 test, 37 dev) with the same evaluation:

| | Baseline (test / dev) | Final (test / dev) |
|---|---|---|
| Citations naming the wrong provision | 3% / 4% | 0 / 0 |
| Completeness, judged against the Act | 0.92 / 0.73 | 0.96 / 0.81 |
| Faithfulness to the retrieved text | 0.93 / 0.96 | 1.00 / 0.98 |
| Unanswerable questions declined | 4 of 4 / 6 of 6 | 4 of 4 / 6 of 6 |
| Recall@1 | 0.85 / 0.68 | 0.85 / 0.68 |
| Recall@3 | 1.00 / 0.94 | 1.00 / 0.81 |
| Cost per question | $0.011 / $0.015 | $0.012 / $0.012 |

Passages are matched to provisions by their text, not their labels, so a mislabelled passage cannot count as a hit. Judges are Claude Haiku 4.5. How the golden set was built: [docs/golden_dataset_methodology.md](docs/golden_dataset_methodology.md).

## Run it

Needs `uv`, an Anthropic API key and an OpenAI API key.

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY and OPENAI_API_KEY

# The Regulation, from EUR-Lex
curl -L -o data/raw/OJ_L_202401689_EN_TXT.pdf \
  "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689"

uv run python -m src.ingestion.pipeline        # build the index (about $0.003 of embeddings)
uv run streamlit run app/main.py               # demo app
uv run uvicorn src.api.main:app --reload       # API
uv run python -m eval.run_eval --split test    # evaluation
uv run python -m eval.compare_runs --split test v1-original v2-final   # compare runs
uv run streamlit run eval/viewer/app.py        # browse runs question by question
uv run pytest                                  # unit tests, no API calls
```

## Files

| Path | What it is |
|---|---|
| `src/ingestion/` | PDF parsing, embedding, ChromaDB |
| `src/retrieval/` | Dense and BM25 search, merging, reranker, recital ranking |
| `src/generation/` | Prompts, the Claude call, citation check, decline detection |
| `src/api/` | FastAPI endpoint and query orchestration |
| `app/` | Streamlit demo |
| `eval/` | Golden set, evaluation runner, LLM judges, run viewer |
| `experiments/` | Saved evaluation runs |

## Limitations

- The corpus is the 2024 text. The Digital Omnibus on AI (Regulation (EU) 2026/1744), in force since 27 July 2026, changes some application dates and is not included; the app says so next to every answer.
- The reranker is trained on English. Another language needs a multilingual reranker and a new evaluation.
- The golden set's labels were checked by Claude against the full text, not yet by a person.
