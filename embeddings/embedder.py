import os
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the local Sentence Transformer model.
        """
        logging.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        logging.info("Model loaded successfully.")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Encodes a list of texts into embeddings.
        """
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Encodes a single query text into an embedding.
        """
        embedding = self.model.encode(query)
        return embedding
