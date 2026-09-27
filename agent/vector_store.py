import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    """
    Create and search a local FAISS vector index for paper chunks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks: list[dict]) -> None:
        """
        Create embeddings for all chunks and build the FAISS index.
        """

        if not chunks:
            raise ValueError("No chunks provided.")

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        self.chunks = chunks

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Return the most relevant chunks for a query.
        """

        if self.index is None:
            raise ValueError("Vector store has not been built.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.chunks)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append(
                {
                    "chunk": self.chunks[index],
                    "score": float(score),
                }
            )

        return results