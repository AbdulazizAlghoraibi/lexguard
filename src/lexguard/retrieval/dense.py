from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from lexguard.schemas.document import DocumentChunk


class DenseRetriever:
    def __init__(
        self,
        chunks: List[DocumentChunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        texts = [c.chunk_text for c in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        self.embeddings = embeddings.astype("float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def query(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)
        return [self.chunks[i] for i in indices[0]]