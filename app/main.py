"""Streamlit demo frontend for the EU AI Act compliance Q&A system.

Run with:
    uv run streamlit run app/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src import config
from src.api.query import answer_query

st.set_page_config(
    page_title="EU AI Act Compliance Q&A",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ EU AI Act Compliance Q&A")
st.caption(
    "Ask questions about the EU AI Act and receive grounded, cited answers "
    "from the official regulatory text."
)
st.caption(f"**Corpus:** {config.CORPUS_DESCRIPTION}")

# ---------------------------------------------------------------------------
# Disclaimer banner
# ---------------------------------------------------------------------------
st.warning(
    "⚠️ **Informational use only.** This tool provides regulatory research support, "
    "not legal advice. Always consult qualified legal counsel for compliance decisions."
)

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
st.divider()

example_queries = [
    "What does ‘AI literacy’ mean in the AI Act?",
    "What quality management obligations apply to providers of high-risk AI systems?",
    "What transparency obligations apply to AI systems that interact with natural persons?",
    "What conformity assessment steps must a provider complete before placing a high-risk AI system on the EU market?",
    # Not in the corpus: the system should decline
    "Which authority supervises the AI Act in Norway?",
]

with st.expander("Example queries"):
    for q in example_queries:
        if st.button(q, key=q):
            st.session_state["query_input"] = q

query = st.text_area(
    "Your question",
    value=st.session_state.get("query_input", ""),
    height=80,
    placeholder="e.g. What are the transparency obligations for high-risk AI systems?",
)

col1, col2 = st.columns([1, 4])
with col1:
    top_k = st.slider(
        "Chunks to retrieve", min_value=1, max_value=10, value=config.TOP_K_RETRIEVAL
    )

submit = st.button("Ask", type="primary", disabled=not query.strip())

# ---------------------------------------------------------------------------
# Response display
# ---------------------------------------------------------------------------
if submit and query.strip():
    with st.spinner("Retrieving relevant provisions and generating answer ..."):
        response = answer_query(query.strip(), top_k=top_k)

    st.divider()

    if not response.answered:
        st.info(
            "**Declined.** The retrieved passages do not answer this question, "
            "so the system says so instead of guessing."
        )
        st.markdown(response.answer or "")
    else:
        # Answer
        st.subheader("Answer")
        st.markdown(response.answer)

        # Metrics strip
        m_cols = st.columns(4)
        m_cols[0].metric("Citations found", len(response.citations))
        m_cols[1].metric(
            "Citations in retrieved text",
            f"{response.citation_verification.accuracy:.0%}"
            if response.citation_verification else "—",
        )
        m_cols[2].metric("Latency", f"{response.latency_ms:.0f}ms")
        m_cols[3].metric("Cost", f"${response.cost_usd:.5f}")

        # Citation check
        if response.citation_verification and response.citations:
            verification = response.citation_verification
            checked = [f"{c} ✓" for c in verification.matched]
            for c in verification.missing:
                if c in verification.cross_referenced:
                    checked.append(f"{c} ↪ named in a retrieved passage")
                else:
                    checked.append(f"{c} ✗ not in the retrieved passages")
            st.markdown("**Citation check:** " + ", ".join(checked))
            st.caption(
                "✓ the cited provision's text was retrieved. ↪ the retrieved text refers to it, "
                "but its own text was not retrieved. ✗ neither."
            )

    if response.retrieved_chunks:
        # Retrieved source chunks
        st.divider()
        st.subheader("Retrieved source passages")
        st.caption("Click to expand and inspect the regulatory text used to generate this answer.")

        for chunk in response.retrieved_chunks:
            label = chunk.article_number or "(unlabelled)"
            if chunk.title:
                label += f" — {chunk.title}"
            score_display = f"Score: {chunk.score:.4f}"

            with st.expander(f"[{chunk.rank + 1}] {label}  |  {score_display}"):
                st.markdown(f"**Source:** {chunk.parent_document}")
                st.markdown(f"**Section type:** {chunk.section_type}")
                st.text(chunk.text)

    # Bad framing warning
    if response.bad_framing_detected:
        st.error(
            "⚠️ **Quality warning:** The response contained phrasing suggesting the model "
            "incorrectly treated the regulatory corpus as user-provided documents. "
            "Please review the answer carefully."
        )
