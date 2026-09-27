# Golden Dataset Methodology

This document describes how the ground-truth evaluation dataset is built for the EU AI Act RAG pipeline. The dataset is the single most important artifact in the project — all retrieval and response metrics are only as meaningful as the quality of this file.

## Version 2 (September 2026)

`golden_dataset.json` is now version 2, built from version 1 by [eval/golden_dataset/build_v2.py](../eval/golden_dataset/build_v2.py). Version 1 is kept as `golden_dataset_v1.json`.

**Why v1 had to change.** The v1 labels came from the pipeline being evaluated. Stage 2 below ran the retriever, and Claude Haiku picked the relevant chunks from its top 5, seeing only the first 600 characters of each. Retrieval was therefore graded against the retriever's own earlier choices, and when the parser mislabelled chunks, the labels inherited the mistakes: 19 of 44 questions pointed at chunks whose label did not match their text. The answer elements were written from the same 600-character excerpts. One answer key said Article 9 "does not provide all four steps" of the risk management system; the Act lists all four, but step (b) starts after character 600. The test split had no unanswerable questions.

**What v2 is.**

- **Labels are provisions**, such as "Article 9" or "Article 3(56)", not chunk hashes. Each was set by reading the question against the full text of the corrected parse, starting from the provisions that v1's chosen chunks actually contain (found by text, see [eval/provisions.py](../eval/provisions.py)). Provision labels survive any change to chunking.
- **Answer elements** were regenerated with Claude Opus 5 from the full text of the gold provisions, then reviewed: one element that inverted Article 50(1) was corrected, and facts the question does not ask about were removed, so a focused answer is not penalised. 126 elements across 44 questions.
- **Ten unanswerable questions** were added, six in dev and four in test, each confirmed absent from the corpus text (for example, which authority supervises the AI Act in Norway).
- **q035** ("Does Article 4 specify the minimum number of training hours?") is now answerable: Article 4 answers it (it does not specify one).
- **Splits** are unchanged for the original 44 questions: 31 dev, 13 test. With the new questions: 37 dev, 17 test.

Labels and answer elements were reviewed by Claude against the full text, not yet by a person.

The rest of this document describes how version 1 was built.

## Overview (version 1)

The pipeline has four stages:

```
candidates_raw.json  →  pre_curated.json  →  golden_dataset.json
      (generate)            (curate)              (select)
```

Each stage is a separate script. The final output is `eval/golden_dataset/golden_dataset.json`.

---

## Stage 1 — Generate candidates

**Script:** `eval/golden_dataset/generate.py`
**Command:** `uv run python -m eval.golden_dataset.generate`
**Output:** `eval/golden_dataset/candidates_raw.json`

Feeds each article chunk to Claude Sonnet and asks it to generate 3–5 candidate queries per article, crossing three taxonomy dimensions:

| Dimension | Values |
|---|---|
| Query type | `factual`, `conceptual`, `procedural`, `cross_reference` |
| Difficulty | `easy`, `medium`, `hard` |
| Persona | `compliance_officer`, `product_manager`, `general` |

A typical run produces ~500 candidates from ~139 article chunks.

**Known limitations:**
- ~20% of articles produce JSON parse errors (the LLM includes unescaped characters from legal text). The script catches these and logs warnings. A tolerant fallback parser recovers most cases on retry.
- Some articles appear multiple times (the structural parser split them across chunk boundaries). This produces duplicate-article candidates that are removed in Stage 2.

**Retrying failed articles:**
```bash
uv run python -m eval.golden_dataset.generate --articles "Article 5" "Article 6" "Article 7"
```
Results are merged into the existing `candidates_raw.json` (deduplicated by query text).

---

## Stage 2 — Curate candidates

**Script:** `eval/golden_dataset/curate.py`
**Command:** `uv run python -m eval.golden_dataset.curate --auto`
**Output:** `eval/golden_dataset/pre_curated.json`

Two modes:

### Basic mode (no API calls)
```bash
uv run python -m eval.golden_dataset.curate
```
Matches `relevant_chunk_ids` by article number only. Fast and free, but the chunk IDs require manual verification because the article-level match may include chunks that don't actually answer the specific query.

### Auto mode (recommended)
```bash
uv run python -m eval.golden_dataset.curate --auto
```
For each candidate query:
1. Runs the actual retrieval pipeline (`search()`) to fetch top-k chunks
2. Calls Claude Haiku to judge which retrieved chunks are genuinely relevant
3. Generates `expected_answer_elements` — 2–4 key facts a correct answer must include
4. Marks `is_unanswerable: true` if no retrieved chunk contains sufficient information

**Estimated cost:** $0.10–0.20 for a full run (~500 candidates, Claude Haiku).

**Quick test run:**
```bash
uv run python -m eval.golden_dataset.curate --auto --max 10
```

After this stage, `pre_curated.json` contains all deduplicated candidates with LLM-verified chunk IDs, answer elements, and preliminary dev/test splits assigned.

---

## Stage 3 — Select final entries

**Script:** `eval/golden_dataset/select.py`
**Command:** `uv run python -m eval.golden_dataset.select`
**Output:** `eval/golden_dataset/golden_dataset.json`

Automatically selects 40–50 entries from `pre_curated.json` using stratified sampling.

### Selection algorithm (three passes)

**Pass 1 — Priority articles**
Guarantees representation of the eight most important EU AI Act provisions:

| Article | Subject |
|---|---|
| Article 5 | Prohibited AI practices |
| Article 6 | High-risk AI system classification |
| Article 9 | Risk management systems |
| Article 10 | Data and data governance |
| Article 13 | Transparency and provision of information |
| Article 14 | Human oversight |
| Article 17 | Quality management system |
| Article 50 | Transparency obligations for certain AI systems |

For each priority article, selects the highest-scoring easy entry and the highest-scoring medium/hard entry.

**Pass 2 — Stratified fill**
Remaining slots are distributed proportionally across query type × difficulty:

| Query type | Target share |
|---|---|
| `factual` | 35% |
| `procedural` | 30% |
| `conceptual` | 20% |
| `cross_reference` | 15% |

| Difficulty | Target share |
|---|---|
| `easy` | 40% |
| `medium` | 40% |
| `hard` | 20% |

Within each stratum, candidates are ranked by a quality score:

| Signal | Points |
|---|---|
| Has ≥1 relevant chunk ID | +3.0 |
| Has >1 relevant chunk ID | +1.0 |
| Has expected answer elements (up to 3) | +1.0 each |
| LLM-verified (not just article-matched) | +2.0 |
| Fallback or no-retrieval note | −2.0 |
| Query text < 40 chars (likely vague) | −1.0 |

**Pass 3 — Unanswerable guarantee**
If no unanswerable query was selected in Passes 1–2, the lowest-scored selected entry is swapped out for the best available unanswerable candidate.

### Split assignment
After selection, entries are assigned to dev/test splits using stratified sampling:
- **70% dev** — used during all experiment iterations
- **30% test** — held out until final evaluation; never used for tuning

Stratification is by `(query_type, difficulty)` to ensure both splits have balanced coverage.

### Options
```bash
# Preview selection without writing
uv run python -m eval.golden_dataset.select --dry-run

# Change target size
uv run python -m eval.golden_dataset.select --target 50
```

---

## Output schema

Each entry in `golden_dataset.json`:

```json
{
  "query_id": "q001",
  "query": "What risk category does real-time biometric identification in public spaces fall under?",
  "query_type": "factual",
  "difficulty": "easy",
  "persona": "compliance_officer",
  "split": "dev",
  "relevant_chunk_ids": ["<sha256_hash>"],
  "partially_relevant_chunk_ids": [],
  "expected_answer_elements": [
    "Real-time remote biometric identification in public spaces is prohibited",
    "Listed under prohibited practices in Article 5",
    "Limited exceptions exist for law enforcement with prior authorisation"
  ],
  "acceptable_citation_targets": ["Article 5(1)(h)"],
  "is_unanswerable": false,
  "reference_answer": "",
  "notes": "llm-verified",
  "source_article": "Article 5"
}
```

**Critical fields for metrics:**

| Field | Used by |
|---|---|
| `relevant_chunk_ids` | Recall@k, MRR, Precision@k |
| `is_unanswerable` | Graceful failure rate |
| `expected_answer_elements` | Manual error analysis |
| `split` | All eval scripts (`--split dev` vs `--split test`) |

---

## Rules for maintaining the dataset

- **Never auto-modify `golden_dataset.json` after manual review is complete.** It is the ground truth. All metric changes should come from pipeline improvements, not from adjusting the ground truth.
- **Never run eval against the test split during development.** The test split is held out for final reporting only.
- **If you add entries, re-run `select.py` with `--dry-run` first** to verify coverage and splits before writing.
- **`relevant_chunk_ids` are content-addressed** (SHA-256 of chunk text). If you re-ingest the corpus with different chunking, these IDs will change and the dataset must be rebuilt.
