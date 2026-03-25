"""Ingestion pipeline orchestrator.

Run with:
    uv run python -m src.ingestion.pipeline
    uv run python -m src.ingestion.pipeline --chunking-strategy fixed-size
    uv run python -m src.ingestion.pipeline --pdf path/to/file.pdf
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src import config
from src.ingestion.chunker import Chunk, chunk_fixed_size, chunk_structural
from src.ingestion.embedder import embed_chunks
from src.ingestion.extractor import extract_pdf
from src.ingestion.parser import parse_document
from src.ingestion.store import store_chunks

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    ingestion_run_id: str
    chunking_strategy: str
    pdf_files: list[str]
    total_chunks: int
    elapsed_seconds: float
    chunks_json_path: str


def run_ingestion(
    pdf_paths: list[Path],
    chunking_strategy: str = "structural",
    collection_name: str = config.CHROMA_COLLECTION_NAME,
    source_url: str = "",
    publication_date: str = "",
) -> IngestionReport:
    """Full ingestion pipeline: PDF → chunks → embeddings → ChromaDB.

    Args:
        pdf_paths: list of PDF files to ingest.
        chunking_strategy: "structural" (baseline) or "fixed-size" (E1).
        collection_name: target ChromaDB collection.
        source_url: official publication URL stored in chunk metadata.
        publication_date: publication date stored in chunk metadata.
    """
    start = time.time()
    all_chunks: list[Chunk] = []

    for pdf_path in pdf_paths:
        logger.info(f"Processing {pdf_path.name} ...")

        pages = extract_pdf(pdf_path)
        units = parse_document(
            pages,
            source_url=source_url,
            publication_date=publication_date,
        )
        logger.info(f"  Parsed {len(units)} structural units from {pdf_path.name}.")

        if chunking_strategy == "fixed-size":
            chunks = chunk_fixed_size(units, ingestion_run_id=config.INGESTION_RUN_ID)
        else:
            chunks = chunk_structural(units, ingestion_run_id=config.INGESTION_RUN_ID)

        logger.info(f"  Produced {len(chunks)} chunks using '{chunking_strategy}'.")
        all_chunks.extend(chunks)

    logger.info(f"Total chunks across all PDFs: {len(all_chunks)}")

    # Persist chunks to JSON (allows re-loading into ChromaDB without re-embedding)
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks_json_path = config.CHUNKS_JSON_PATH
    with open(chunks_json_path, "w") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2)
    logger.info(f"Saved chunks to {chunks_json_path}")

    # Embed and store
    logger.info("Embedding chunks ...")
    chunks_with_embeddings = embed_chunks(all_chunks)

    logger.info("Storing in ChromaDB ...")
    store_chunks(chunks_with_embeddings, collection_name=collection_name)

    elapsed = time.time() - start
    report = IngestionReport(
        ingestion_run_id=config.INGESTION_RUN_ID,
        chunking_strategy=chunking_strategy,
        pdf_files=[p.name for p in pdf_paths],
        total_chunks=len(all_chunks),
        elapsed_seconds=round(elapsed, 1),
        chunks_json_path=str(chunks_json_path),
    )
    logger.info(
        f"Ingestion complete. {len(all_chunks)} chunks in {elapsed:.1f}s. "
        f"Run ID: {config.INGESTION_RUN_ID}"
    )
    return report


def _find_pdfs(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.pdf"))


def main() -> None:
    parser = argparse.ArgumentParser(description="EU AI Act ingestion pipeline")
    parser.add_argument(
        "--pdf",
        type=Path,
        nargs="*",
        help="Specific PDF file(s) to ingest. Defaults to all PDFs in data/raw/.",
    )
    parser.add_argument(
        "--chunking-strategy",
        choices=["structural", "fixed-size"],
        default="structural",
        help="Chunking strategy (default: structural).",
    )
    parser.add_argument(
        "--collection",
        default=config.CHROMA_COLLECTION_NAME,
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--source-url",
        default="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
        help="Official source URL stored in chunk metadata.",
    )
    parser.add_argument(
        "--publication-date",
        default="2024-07-12",
        help="Publication date stored in chunk metadata (YYYY-MM-DD).",
    )
    args = parser.parse_args()

    pdf_paths = args.pdf or _find_pdfs(config.DATA_RAW_DIR)
    if not pdf_paths:
        logger.error(
            f"No PDFs found. Place EU AI Act PDFs in {config.DATA_RAW_DIR} "
            "or pass --pdf <path>."
        )
        raise SystemExit(1)

    report = run_ingestion(
        pdf_paths=pdf_paths,
        chunking_strategy=args.chunking_strategy,
        collection_name=args.collection,
        source_url=args.source_url,
        publication_date=args.publication_date,
    )
    print(f"\nIngestion report: {asdict(report)}")


if __name__ == "__main__":
    main()
