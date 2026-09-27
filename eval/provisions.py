"""Which provisions of the Act a piece of text actually contains.

The evaluation's ground truth for retrieval and citations. It is built from the
corrected parse (corpus v2) and matches text, not labels: a passage contains a
provision when it shares a run of words that occurs in that provision and
nowhere else in the Act. A chunk from any corpus version, including the original
parse whose labels were wrong, is judged by what it says rather than by what it
is called. Trusting labels is how the original evaluation missed the parser bug.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from src import config
from src.generation.citations import parse_reference

V2_CHUNKS_PATH: Path = config.DATA_PROCESSED_DIR / "chunks_v2.json"

_WINDOW = 12  # words per matching window


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class ProvisionIndex:
    """Maps distinctive 12-word windows of the Act's text to the provision they belong to."""

    def __init__(self, chunks_path: Path = V2_CHUNKS_PATH) -> None:
        with open(chunks_path) as f:
            chunks = json.load(f)
        self.text: dict[str, str] = {c["article_number"]: c["text"] for c in chunks}
        self.title: dict[str, str] = {c["article_number"]: c["title"] for c in chunks}

        owners: dict[tuple[str, ...], set[str]] = {}
        for label, text in self.text.items():
            words = _words(text)
            if re.fullmatch(r"Article \d+\(\d+\)", label):
                words = words[3:]  # skip the "Article 3 Definitions" heading every definition repeats
            for i in range(len(words) - _WINDOW + 1):
                owners.setdefault(tuple(words[i : i + _WINDOW]), set()).add(label)
        # Windows shared by several provisions (boilerplate) prove nothing
        self._owner = {w: next(iter(labels)) for w, labels in owners.items() if len(labels) == 1}

    def provisions_in(self, text: str) -> list[str]:
        """Provisions whose text appears in `text`, in order of first appearance."""
        words = _words(text)
        first_seen: dict[str, int] = {}
        for i in range(len(words) - _WINDOW + 1):
            label = self._owner.get(tuple(words[i : i + _WINDOW]))
            if label and label not in first_seen:
                first_seen[label] = i
        return sorted(first_seen, key=first_seen.get)


@lru_cache(maxsize=1)
def get_index() -> ProvisionIndex:
    return ProvisionIndex()


def refers_to(label: str, other: str) -> bool:
    """True if two provision labels overlap, e.g. "Article 3" and "Article 3(56)"."""
    a, b = parse_reference(label), parse_reference(other)
    if a is None or b is None:
        return label == other
    return a.overlaps(b)


def contains_any(provisions: list[str], targets: list[str]) -> bool:
    """True if any provision in `provisions` overlaps any provision in `targets`."""
    return any(refers_to(p, t) for p in provisions for t in targets)
