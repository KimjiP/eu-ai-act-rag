"""Streamlit eval viewer for manual error analysis and annotation.

Run with:
    uv run streamlit run eval/viewer/app.py

Reads the files written by eval/run_eval.py (experiments/eval_<tag>_<split>.json).

Features:
- Summary of a run, or of two runs side by side (before/after a change)
- Per question: gold provisions and answer facts, what was retrieved and which
  provisions those passages really contain, the answer, the citation check,
  and the judges' verdicts
- Annotation panel: tag failure class, add notes, save to error_analysis.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from src import config

EXPERIMENTS_DIR = config.EXPERIMENTS_DIR
ERROR_ANALYSIS_PATH = EXPERIMENTS_DIR / "error_analysis.json"

# Failure taxonomy from PRD Section 6.5
FAILURE_CLASSES = [
    "none",
    "retrieval_miss",
    "retrieval_ranking_issue",
    "context_overload",
    "unsupported_generation",
    "citation_mismatch",
    "scope_violation",
    "bad_framing",
]

SUMMARY_ROWS = [
    ("Recall@1", "retrieval", "recall_at_1"),
    ("Recall@3", "retrieval", "recall_at_3"),
    ("Recall@5", "retrieval", "recall_at_5"),
    ("MRR", "retrieval", "mrr"),
    ("Citations whose text was retrieved", "answers", "citations_grounded"),
    ("Citations named in retrieved text (cross-reference)", "answers", "citations_cross_reference"),
    ("Citations unsupported", "answers", "citations_unsupported"),
    ("Declined, unanswerable (should be 1.0)", "answers", "unanswerable_declined"),
    ("Declined, answerable (should be 0.0)", "answers", "answerable_declined"),
    ("Completeness vs the Act (judge)", "judges", "completeness"),
    ("Faithfulness to retrieved text (judge)", "judges", "faithfulness"),
    ("Answer relevance (judge)", "judges", "answer_relevance"),
    ("Avg context sent to the LLM (chars)", "answers", "avg_context_chars"),
    ("Avg cost per question (USD)", "answers", "avg_cost_usd"),
]


def load_run(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_golden() -> dict[str, dict]:
    if not config.GOLDEN_DATASET_PATH.exists():
        return {}
    with open(config.GOLDEN_DATASET_PATH) as f:
        return {q["query_id"]: q for q in json.load(f)}


def load_error_analysis() -> dict[str, dict]:
    if not ERROR_ANALYSIS_PATH.exists():
        return {}
    with open(ERROR_ANALYSIS_PATH) as f:
        data = json.load(f)
    return {r["query_id"]: r for r in data}


def save_error_analysis(annotations: dict[str, dict]) -> None:
    ERROR_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_ANALYSIS_PATH, "w") as f:
        json.dump(list(annotations.values()), f, indent=2)


def question_status(record: dict) -> str:
    answer = record["answer"]
    if record["is_unanswerable"]:
        return "✓" if answer["declined"] else "✗"
    if answer["declined"]:
        return "✗"
    retrieved = record.get("retrieval", {}).get("recall_at_3", 0) == 1.0
    grounded = all(c["grounded"] for c in answer["citations"])
    return "✓" if retrieved and grounded else "~"


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="EU AI Act RAG — Eval Viewer", layout="wide")
st.title("EU AI Act RAG — Evaluation Viewer")

run_files = sorted(EXPERIMENTS_DIR.glob("eval_*.json")) if EXPERIMENTS_DIR.exists() else []
if not run_files:
    st.warning(
        f"No evaluation runs found in `{EXPERIMENTS_DIR}/`. "
        "Run `uv run python -m eval.run_eval --split dev --tag <name>` first."
    )
    st.stop()

st.sidebar.header("Runs")
names = [f.name for f in run_files]
file_a = st.sidebar.selectbox("Run A", names)
compare = st.sidebar.checkbox("Compare with a second run")
file_b = st.sidebar.selectbox("Run B", [n for n in names if n != file_a]) if compare else None

runs = {"A": load_run(EXPERIMENTS_DIR / file_a)}
if file_b:
    runs["B"] = load_run(EXPERIMENTS_DIR / file_b)
golden = load_golden()
annotations = load_error_analysis()

# Summary table
st.subheader("Summary")
header = "| Metric | " + " | ".join(f"{k}: {r['tag']} ({r['split']})" for k, r in runs.items()) + " |"
lines = [header, "|---|" + "---|" * len(runs)]
for label, section, key in SUMMARY_ROWS:
    values = [str(r["summary"].get(section, {}).get(key, "—")) for r in runs.values()]
    lines.append(f"| {label} | " + " | ".join(values) + " |")
st.markdown("\n".join(lines))
for key, run in runs.items():
    c = run["config"]
    st.caption(
        f"{key}: corpus {c['corpus_version']}, rerank candidates {c['rerank_candidates']}, "
        f"top_k {c['top_k']}, {c['llm_model']}, judge {c['judge_model']}, run {run['run_at']}"
    )

# Question selector
records_a = {r["query_id"]: r for r in runs["A"]["per_query"]}
query_ids = list(records_a)
selected = st.sidebar.selectbox(
    "Question",
    query_ids,
    format_func=lambda q: f"{question_status(records_a[q])} {q} — {records_a[q]['query'][:50]}",
)
st.sidebar.caption("✓ passed, ~ partly, ✗ failed (retrieval@3, citations, decline)")

st.divider()
item = golden.get(selected, {})
record_a = records_a[selected]
st.markdown(f"### {selected}: {record_a['query']}")
if record_a["is_unanswerable"]:
    st.info("Not answerable from the corpus. The right behaviour is to decline.")
else:
    st.markdown(f"**Gold provisions:** {', '.join(record_a['gold_provisions'])}")
    with st.expander("Facts a complete answer must state"):
        for element in item.get("expected_answer_elements", []):
            st.markdown(f"- {element}")


def render_record(container, record: dict | None, label: str) -> None:
    with container:
        st.markdown(f"#### {label}")
        if record is None:
            st.warning("This question is not in the selected run.")
            return
        if "retrieval" in record:
            r = record["retrieval"]
            st.caption(f"Recall@1 {r['recall_at_1']}, @3 {r['recall_at_3']}, MRR {r['mrr']:.2f}")
        st.markdown("**Retrieved (top 5)**")
        for p in record["retrieved"]:
            hit = p["relevant"]
            more =f" … +{p['contains_count'] - len(p['contains'])} more" if p["contains_count"] > len(p["contains"]) else ""
            st.markdown(
                f"{'✅' if hit else '▫️'} [{p['rank']}] labelled **{p['label']}**, contains "
                f"{', '.join(p['contains']) or '—'}{more} ({p['chars']:,} chars)"
            )

        answer = record["answer"]
        st.markdown("**Answer**")
        if answer["declined"]:
            st.info("Declined")
        st.markdown(answer["text"])
        if answer["citations"]:
            marks = {"retrieved": "✓", "cross_reference": "↪ cross-reference", "unsupported": "✗"}
            st.markdown(
                "**Citation check:** "
                + ", ".join(f"{c['citation']} {marks[c['status']]}" for c in answer["citations"])
            )
            st.caption("✓ text retrieved, ↪ named in the retrieved text but not retrieved, ✗ unsupported")
        for name, verdict in record.get("judges", {}).items():
            st.markdown(f"**{name}: {verdict['score']:.2f}** — {verdict.get('reasoning', '')}")
        st.caption(f"Cost ${answer['cost_usd']:.4f}, latency {answer['latency_ms']:.0f} ms")


if "B" in runs:
    col_a, col_b = st.columns(2)
    records_b = {r["query_id"]: r for r in runs["B"]["per_query"]}
    render_record(col_a, record_a, f"A: {runs['A']['tag']}")
    render_record(col_b, records_b.get(selected), f"B: {runs['B']['tag']}")
else:
    render_record(st.container(), record_a, runs["A"]["tag"])

# ---------------------------------------------------------------------------
# Annotation panel
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Error analysis annotation")

existing = annotations.get(selected, {"query_id": selected, "failure_class": "none", "notes": ""})
failure_class = st.selectbox(
    "Failure class",
    FAILURE_CLASSES,
    index=FAILURE_CLASSES.index(existing.get("failure_class", "none")),
)
notes = st.text_area("Notes", value=existing.get("notes", ""), height=80)

if st.button("Save annotation"):
    annotations[selected] = {"query_id": selected, "failure_class": failure_class, "notes": notes}
    save_error_analysis(annotations)
    st.success("Annotation saved.")

st.sidebar.divider()
st.sidebar.subheader("Annotation summary")
if annotations:
    from collections import Counter

    counts = Counter(v.get("failure_class", "none") for v in annotations.values())
    for cls, count in counts.most_common():
        st.sidebar.write(f"{cls}: {count}")
    st.sidebar.write(f"Total annotated: {len(annotations)}")
