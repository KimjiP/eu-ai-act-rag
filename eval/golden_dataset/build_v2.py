"""Build golden dataset v2 from v1 (September 2026 audit).

What changed from v1, and why:
  - Labels are provisions ("Article 9", "Article 3(56)"), not chunk hashes. The v1
    labels were picked by Claude Haiku from the retriever's own top 5, seeing only
    the first 600 characters of each chunk. Retrieval recall was therefore measured
    against the retriever's earlier choices, and 19 of 44 labels pointed at chunks
    the original parser had mislabelled. Each label below was set by reading the
    question against the full text of the corrected parse.
  - Expected answer elements are regenerated from the full text of those provisions
    (v1's were written from the 600-character excerpts).
  - Ten questions the corpus cannot answer were added, six in dev and four in test,
    so decline behaviour is measured on both splits. v1 had one, in dev.
  - q035 ("Does Article 4 specify the minimum number of training hours?") is now
    answerable: Article 4 answers it (it does not specify one).

Labels were reviewed by Claude against the Act's text, not yet by a person.

Run with:
    uv run python -m eval.golden_dataset.build_v2
Answer elements are generated once with Claude, then reviewed in golden_dataset.json:
one inverted statement of Article 50(1) was corrected, and facts the question does not
ask about were removed so a focused answer is not penalised. Re-running keeps the
edited elements; delete a question's `expected_answer_elements` to regenerate it.
"""

import json
import logging

import anthropic
from pydantic import BaseModel

from src import config
from eval.provisions import get_index

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

V1_PATH = config.EVAL_DIR / "golden_dataset" / "golden_dataset_v1.json"
V2_PATH = config.GOLDEN_DATASET_PATH
ANSWER_KEY_MODEL = "claude-opus-5"
REVIEW_NOTE = "labels reviewed against the full text (Claude, 2026-09-28); human spot-check pending"

# Provisions that answer each v1 question. The first is the primary one.
GOLD: dict[str, list[str]] = {
    "q001": ["Article 9"],
    "q002": ["Article 9"],
    "q003": ["Article 41"],
    "q004": ["Article 41"],
    "q005": ["Article 5"],
    "q006": ["Article 5"],
    "q007": ["Article 71"],
    "q008": ["Article 71", "Article 49"],
    "q009": ["Article 14"],
    "q010": ["Article 14"],
    "q011": ["Article 17"],
    "q012": ["Article 17"],
    "q013": ["Article 13"],
    "q014": ["Article 15", "Article 13", "Annex IV"],
    "q015": ["Article 50"],
    "q016": ["Article 50"],
    "q017": ["Article 12"],
    "q018": ["Article 57"],
    "q019": ["Article 1"],
    "q020": ["Article 25", "Article 3(3)"],
    "q021": ["Article 3(3)", "Article 3(4)"],
    "q022": ["Article 34", "Article 43", "Annex VII"],
    "q023": ["Article 40"],
    "q024": ["Article 2"],
    "q025": ["Article 9", "Article 72"],
    "q026": ["Article 17"],
    "q027": ["Article 2"],
    "q028": ["Article 23"],
    "q029": ["Article 3(8)"],
    "q030": ["Article 4", "Article 3(56)"],
    "q031": ["Article 22"],
    "q032": ["Article 97"],
    "q033": ["Article 1"],
    "q034": ["Article 1"],
    "q035": ["Article 4"],
    "q036": ["Article 8"],
    "q037": ["Article 4", "Article 26"],
    "q038": ["Article 12", "Article 19", "Article 26"],
    "q039": ["Article 15", "Article 13", "Annex IV"],
    "q040": ["Article 17"],
    "q041": ["Article 19", "Article 26"],
    "q042": ["Article 22", "Article 54"],
    "q043": ["Article 11", "Article 8"],
    "q044": ["Article 11"],
}

# Questions whose answer is not in the corpus. The right behaviour is to decline.
UNANSWERABLE: list[tuple[str, str, str]] = [
    ("u01", "dev", "Which companies have been fined under the AI Act so far, and how large were the fines?"),
    ("u02", "dev", "Who is the current head of the European AI Office?"),
    ("u03", "dev", "When will Norway incorporate the AI Act into the EEA Agreement?"),
    ("u04", "dev", "Which notified bodies have been designated under the AI Act so far?"),
    (
        "u05",
        "dev",
        "What did the Commission's 2025 guidelines on the AI system definition conclude "
        "about logistic regression models?",
    ),
    ("u06", "dev", "How many high-risk AI systems are currently registered in the EU database?"),
    (
        "u07",
        "test",
        "Which national authority has Norway designated as its market surveillance authority "
        "for the AI Act?",
    ),
    ("u08", "test", "How much does it cost to register a high-risk AI system in the EU database?"),
    ("u09", "test", "What changes did the 2026 Digital Omnibus make to the AI Act?"),
    (
        "u10",
        "test",
        "Which AI systems did the Commission add to Annex III in its first delegated act "
        "amending the list?",
    ),
]

ANSWER_KEY_PROMPT = """You write answer keys for an evaluation of a question-answering system over the EU AI Act.

Given a question and the full text of the provisions that answer it, list the 2 to 4 facts a correct and complete answer must state.

Rules:
- Each fact is one short sentence taken from the provision text, not from general knowledge.
- End each fact with the provision it comes from in parentheses, as precisely as the text allows, e.g. "(Article 9(2)(a))" or "(Article 3(56))".
- If the provisions show that the answer is "no", or that something is not specified, state that as a fact.
- Cover what the question asks. Do not add facts it does not ask about.

Question: {question}

Provisions:
{provisions}"""


class AnswerKey(BaseModel):
    elements: list[str]


def _answer_elements(client: anthropic.Anthropic, question: str, gold: list[str]) -> list[str]:
    index = get_index()
    provisions = "\n\n".join(f"[{label}]\n{index.text[label]}" for label in gold)
    response = client.messages.parse(
        model=ANSWER_KEY_MODEL,
        max_tokens=16000,
        output_config={"effort": "low"},
        messages=[
            {
                "role": "user",
                "content": ANSWER_KEY_PROMPT.format(question=question, provisions=provisions),
            }
        ],
        output_format=AnswerKey,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        logger.warning(f"No answer key generated ({response.stop_reason}): {question[:60]!r}")
        return []
    return response.parsed_output.elements


def main() -> None:
    with open(V1_PATH) as f:
        v1 = json.load(f)
    previous = {}
    if V2_PATH.exists():
        with open(V2_PATH) as f:
            previous = {item["query_id"]: item for item in json.load(f)}

    index = get_index()
    missing = [p for gold in GOLD.values() for p in gold if p not in index.text]
    if missing:
        raise ValueError(f"Gold provisions not in the corpus: {missing}")

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    dataset = []

    for item in v1:
        query_id = item["query_id"]
        gold = GOLD[query_id]
        cached = previous.get(query_id, {})
        elements = cached.get("expected_answer_elements") if cached.get("gold_provisions") == gold else None
        if not elements:
            logger.info(f"Generating answer key for {query_id}")
            elements = _answer_elements(client, item["query"], gold)
        dataset.append(
            {
                "query_id": query_id,
                "query": item["query"],
                "query_type": item["query_type"],
                "difficulty": item["difficulty"],
                "persona": item["persona"],
                "split": item["split"],
                "is_unanswerable": False,
                "gold_provisions": gold,
                "expected_answer_elements": elements,
                "notes": REVIEW_NOTE,
            }
        )

    for query_id, split, question in UNANSWERABLE:
        dataset.append(
            {
                "query_id": query_id,
                "query": question,
                "query_type": "out_of_corpus",
                "difficulty": "medium",
                "persona": "general",
                "split": split,
                "is_unanswerable": True,
                "gold_provisions": [],
                "expected_answer_elements": [],
                "notes": "added 2026-09-28; answer confirmed absent from the corpus text",
            }
        )

    with open(V2_PATH, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    n_unanswerable = sum(item["is_unanswerable"] for item in dataset)
    logger.info(
        f"Wrote {len(dataset)} questions ({n_unanswerable} unanswerable) to {V2_PATH}"
    )


if __name__ == "__main__":
    main()
