"""Central configuration for the EU AI Act RAG pipeline.

Every tuneable value lives here. Changing a variable for an experiment
means changing one line in this file only — never hard-code values elsewhere.
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
TOP_K_RETRIEVAL: int = 5
# ChromaDB returns L2 distances; lower = more similar.
# Results with distance > this threshold are treated as low-confidence.
CONFIDENCE_THRESHOLD: float = 1.5
# "dense" | "hybrid"
RETRIEVAL_STRATEGY: str = "hybrid"
# BM25 / dense fusion weight: 0.0 = BM25 only, 1.0 = dense only
HYBRID_ALPHA: float = 0.5
RERANKER_ENABLED: bool = True
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PROMPT_VERSION: str = "v1"

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME: str = "eu_ai_act_v1"

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
CHUNKS_JSON_PATH: Path = DATA_PROCESSED_DIR / "chunks_v1.json"
QUERY_LOG_PATH: Path = EXPERIMENTS_DIR / "query_log.jsonl"
EXPERIMENT_LOG_PATH: Path = EXPERIMENTS_DIR / "experiment_log.json"

# Ensure runtime directories exist
for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, CHROMA_PERSIST_DIR, EXPERIMENTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
