"""Central configuration for the EU AI Act RAG pipeline.

Every tuneable value lives here. Changing a variable for an experiment
means changing one line in this file only — never hard-code values elsewhere.

Three values can also be set from the environment, so the evaluation can run the
original system and each step of the fix from the same code:
    RAG_CORPUS_VERSION=v1     the original parse, whose chunk labels are known to be wrong
    RAG_RERANK_CANDIDATES=0   rerank only the top_k results, as originally built
    RAG_RECITAL_PENALTY=0     rank recitals like any other passage
"""

import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API Keys (read lazily — missing keys raise only when the key is actually used,
# not at import time, so unit tests that don't call external APIs can run freely)
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "text-embedding-3-small"
LLM_MODEL: str = "claude-sonnet-4-5"

# ---------------------------------------------------------------------------
# LLM Generation
# ---------------------------------------------------------------------------
LLM_TEMPERATURE: float = 0.1
LLM_MAX_TOKENS: int = 1024

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# Chunks passed to the LLM. E3 found 3 better than 5 (citation completeness, cost).
TOP_K_RETRIEVAL: int = 3
# "dense" | "hybrid"
RETRIEVAL_STRATEGY: str = "hybrid"
# Not used by the RRF fusion in hybrid.py, which weights both rankings equally.
HYBRID_ALPHA: float = 0.5
RERANKER_ENABLED: bool = True
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Hybrid-search candidates the reranker re-scores before keeping top_k.
# 0 = rerank only the top_k results (the original behaviour).
RERANK_CANDIDATES: int = int(os.environ.get("RAG_RERANK_CANDIDATES", "20"))
# Subtracted from recitals' reranker scores, so the operative text (articles,
# definitions, annexes) ranks first. Recitals are short, plain-language and
# fully visible to the 512-token reranker, so they outscored the articles that
# carry the obligations. Tuned on the dev split (E9); 0 disables it.
RECITAL_PENALTY: float = float(os.environ.get("RAG_RECITAL_PENALTY", "3.0"))

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PROMPT_VERSION: str = "v1"

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
# v2: corrected structural parse (September 2026). v1: the original parse, kept
# only so the evaluation can measure the original system.
CORPUS_VERSION: str = os.environ.get("RAG_CORPUS_VERSION", "v2")
CHROMA_COLLECTION_NAME: str = f"eu_ai_act_{CORPUS_VERSION}"
# Which legal text the corpus holds, shown next to every answer
CORPUS_DESCRIPTION: str = (
    "Regulation (EU) 2024/1689 as published in the Official Journal on 12 July 2024. "
    "Later amendments, including Regulation (EU) 2026/1744 (the Digital Omnibus on AI), "
    "are not included."
)

# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------
EXPERIMENT_ID: str = "E0"
INGESTION_RUN_ID: str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).parent.parent
DATA_RAW_DIR: Path = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = ROOT_DIR / "data" / "processed"
CHROMA_PERSIST_DIR: Path = ROOT_DIR / "data" / "chroma"
EXPERIMENTS_DIR: Path = ROOT_DIR / "experiments"
EVAL_DIR: Path = ROOT_DIR / "eval"
GOLDEN_DATASET_PATH: Path = EVAL_DIR / "golden_dataset" / "golden_dataset.json"
CHUNKS_JSON_PATH: Path = DATA_PROCESSED_DIR / f"chunks_{CORPUS_VERSION}.json"
QUERY_LOG_PATH: Path = EXPERIMENTS_DIR / "query_log.jsonl"
EXPERIMENT_LOG_PATH: Path = EXPERIMENTS_DIR / "experiment_log.json"

# Ensure runtime directories exist
for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, CHROMA_PERSIST_DIR, EXPERIMENTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
