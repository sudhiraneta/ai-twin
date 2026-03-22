from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class EmbeddingEngine:
    """Generates embeddings using sentence-transformers (local, free, private)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self._model = None
        self._model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"Loading embedding model: {self._model_name}...")
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]
