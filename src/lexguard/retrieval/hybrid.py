from typing import List, Tuple
import numpy as np

from lexguard.schemas.document import DocumentChunk
from lexguard.retrieval.bm25 import BM25Retriever
from lexguard.retrieval.dense import DenseRetriever


class HybridRetriever:
    def __init__(
        self,
        chunks: List[DocumentChunk],
        bm25_weight: float = 0.45,
        dense_weight: float = 0.55,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.chunks = chunks
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        self.bm25_retriever = BM25Retriever(chunks)
        self.dense_retriever = DenseRetriever(chunks, model_name=model_name)

    def _tokenize(self, text: str):
        return text.lower().split()

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        scores = np.array(scores, dtype=float)
        if len(scores) == 0:
            return scores
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s < 1e-8:
            return np.ones_like(scores)
        return (scores - min_s) / (max_s - min_s)

    def query(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        tokenized_query = self._tokenize(query)

        bm25_scores = np.array(
            self.bm25_retriever.bm25.get_scores(tokenized_query),
            dtype=float
        )

        query_embedding = self.dense_retriever.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        dense_scores, _ = self.dense_retriever.index.search(
            query_embedding,
            len(self.chunks)
        )
        dense_scores = dense_scores[0].astype(float)

        bm25_norm = self._normalize(bm25_scores)
        dense_norm = self._normalize(dense_scores)

        final_scores = (
            self.bm25_weight * bm25_norm
            + self.dense_weight * dense_norm
        )

        ranked_indices = np.argsort(final_scores)[::-1][:top_k]
        return [self.chunks[i] for i in ranked_indices]

    def debug_query(self, query: str, top_k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        tokenized_query = self._tokenize(query)

        bm25_scores = np.array(
            self.bm25_retriever.bm25.get_scores(tokenized_query),
            dtype=float
        )

        query_embedding = self.dense_retriever.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        dense_scores, _ = self.dense_retriever.index.search(
            query_embedding,
            len(self.chunks)
        )
        dense_scores = dense_scores[0].astype(float)

        bm25_norm = self._normalize(bm25_scores)
        dense_norm = self._normalize(dense_scores)
        final_scores = (
            self.bm25_weight * bm25_norm
            + self.dense_weight * dense_norm
        )

        ranked_indices = np.argsort(final_scores)[::-1][:top_k]
        return [(self.chunks[i], float(final_scores[i])) for i in ranked_indices]