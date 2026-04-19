from rank_bm25 import BM25Okapi
from typing import List
from lexguard.schemas.document import DocumentChunk


class BM25Retriever:
    def __init__(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.corpus = [self._tokenize(c.chunk_text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)

    def _tokenize(self, text: str):
        return text.lower().split()

    def query(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [chunk for chunk, _ in ranked[:top_k]]