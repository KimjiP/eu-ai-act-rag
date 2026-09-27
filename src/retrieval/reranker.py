"""Cross-encoder reranker (Experiment E5).

Uses a sentence-transformers cross-encoder to re-score (query, chunk) pairs.
Only activated when config.RERANKER_ENABLED = True.
"""

import logging

from src import config
from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = config.RERANKER_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading cross-encoder reranker: {model_name}")
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = config.TOP_K_RETRIEVAL,
    ) -> list[RetrievalResult]:
        """Re-score and re-rank results using the cross-encoder.

        Returns up to top_k results ordered by cross-encoder score (highest first).
        """
        if not results:
            return results

        pairs = [(query, r.text) for r in results]
        scores = self._model.predict(pairs, show_progress_bar=False)

        scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

        reranked: list[RetrievalResult] = []
        for new_rank, (result, score) in enumerate(scored[:top_k]):
            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=float(score),
                    article_number=result.article_number,
                    section_type=result.section_type,
                    title=result.title,
                    parent_document=result.parent_document,
                    rank=new_rank,
                )
            )

        logger.debug(f"Reranker re-scored {len(results)} → returned {len(reranked)}.")
        return reranked


_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker
