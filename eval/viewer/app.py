"""Streamlit eval viewer for manual error analysis and annotation.

Run with:
    uv run streamlit run eval/viewer/app.py

Features:
- Per-query view: query, retrieved chunks (with scores), generated answer, ground truth
- Side-by-side mode: compare two experiment result files before/after a change
- Annotation panel: tag failure class, add notes, save to error_analysis.json
"""

import json
from pathlib import Path

import streamlit as st

from src import config

GOLDEN_DATASET_PATH = config.GOLDEN_DATASET_PATH
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


def load_golden_dataset() -> list[dict]:
    if not GOLDEN_DATASET_PATH.exists():
        return []
    with open(GOLDEN_DATASET_PATH) as f:
        return json.load(f)


def load_results(path: Path) -> dict[str, dict]:
    """Load a response results JSON, keyed by query_id."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {r["query_id"]: r for r in data.get("per_query", [])}


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


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="EU AI Act RAG — Eval Viewer", layout="wide")
st.title("EU AI Act RAG — Evaluation Viewer")

# Sidebar: file selection and mode
st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["Single experiment", "Side-by-side comparison"])

result_files = sorted(EXPERIMENTS_DIR.glob("response_*.json")) if EXPERIMENTS_DIR.exists() else []
file_options = {f.name: f for f in result_files}

if not file_options:
    st.warning(
        f"No response result files found in `{EXPERIMENTS_DIR}/`. "
        "Run `uv run python eval/metrics/run_response_eval.py` first."
    )
    st.stop()

selected_file_a = st.sidebar.selectbox("Results file (A)", list(file_options.keys()))
results_a = load_results(file_options[selected_file_a])

results_b: dict[str, dict] = {}
if mode == "Side-by-side comparison":
    selected_file_b = st.sidebar.selectbox(
        "Results file (B)", [f for f in file_options if f != selected_file_a]
    )
    if selected_file_b:
        results_b = load_results(file_options[selected_file_b])

# Filter by split
split_filter = st.sidebar.selectbox("Split", ["all", "dev", "test"])

# Load data
golden = load_golden_dataset()
annotations = load_error_analysis()

if split_filter != "all":
    golden = [q for q in golden if q.get("split") == split_filter]

if not golden:
    st.warning("No golden dataset entries found. Run ingestion and generate the golden dataset first.")
    st.stop()

# Query selector
query_ids = [q["query_id"] for q in golden]
query_labels = [f"{q['query_id']} — {q['query'][:60]}..." for q in golden]
selected_idx = st.sidebar.selectbox("Query", range(len(query_ids)), format_func=lambda i: query_labels[i])

item = golden[selected_idx]
query_id = item["query_id"]

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1]) if mode == "Side-by-side comparison" else [st.container(), None]

def render_query_panel(container, item: dict, results: dict[str, dict], label: str = "") -> None:
    with container:
        if label:
            st.subheader(label)

        st.markdown(f"**Query ({item.get('difficulty','?')} / {item.get('query_type','?')} / {item.get('persona','?')})**")
        st.info(item["query"])

        # Ground truth
        with st.expander("Ground truth", expanded=False):
            st.markdown(f"**Expected elements:** {item.get('expected_answer_elements', [])}")
            st.markdown(f"**Acceptable citations:** {item.get('acceptable_citation_targets', [])}")
            st.markdown(f"**Unanswerable:** {item.get('is_unanswerable', False)}")
            if item.get("reference_answer"):
                st.markdown(f"**Reference answer:** {item['reference_answer']}")

        result = results.get(query_id)
        if not result:
            st.warning("No result found for this query in the selected results file.")
            return

        # Metrics strip
        m_cols = st.columns(4)
        m_cols[0].metric("Answered", "✓" if result.get("answered") else "✗")
        m_cols[1].metric("Citation acc.", f"{result.get('citation_accuracy', 0):.2f}")
        m_cols[2].metric("Bad framing", "⚠️ YES" if result.get("bad_framing") else "OK")
        m_cols[3].metric("Latency", f"{result.get('latency_ms', 0):.0f}ms")

        # Answer
        st.markdown("**Generated answer:**")
        if result.get("answered") and result.get("answer"):
            st.markdown(result["answer"])
        else:
            st.warning("System declined to answer (graceful failure).")


render_query_panel(col_left, item, results_a, label="Experiment A" if mode == "Side-by-side comparison" else "")
if mode == "Side-by-side comparison" and col_right and results_b:
    render_query_panel(col_right, item, results_b, label="Experiment B")

# ---------------------------------------------------------------------------
# Annotation panel
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Error analysis annotation")

existing = annotations.get(query_id, {
    "query_id": query_id,
    "failure_class": "none",
    "notes": "",
})

failure_class = st.selectbox(
    "Failure class",
    FAILURE_CLASSES,
    index=FAILURE_CLASSES.index(existing.get("failure_class", "none")),
)
notes = st.text_area("Notes", value=existing.get("notes", ""), height=80)

if st.button("Save annotation"):
    annotations[query_id] = {
        "query_id": query_id,
        "failure_class": failure_class,
        "notes": notes,
    }
    save_error_analysis(annotations)
    st.success("Annotation saved.")

# Annotation summary in sidebar
st.sidebar.divider()
st.sidebar.subheader("Annotation summary")
if annotations:
    from collections import Counter
    counts = Counter(v.get("failure_class", "none") for v in annotations.values())
    for cls, count in counts.most_common():
        st.sidebar.write(f"{cls}: {count}")
    st.sidebar.write(f"Total annotated: {len(annotations)} / {len(golden)}")
