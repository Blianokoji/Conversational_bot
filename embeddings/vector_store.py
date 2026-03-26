import os
import json
import faiss
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class FAISSVectorStore:
    def __init__(self, index_path: str = None, metadata_path: str = None):
        self.index = None
        self.metadata = []
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        if index_path and os.path.exists(index_path) and metadata_path and os.path.exists(metadata_path):
            self.load()

    def build_index(self, embeddings: np.ndarray, metadata: list[dict]):
        """
        Builds the FAISS index from a numpy array of embeddings and their corresponding metadata.
        """
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.metadata = metadata
        logging.info(f"FAISS index built with {self.index.ntotal} vectors of dimension {dimension}.")

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Searches the FAISS index for the top_k most similar vectors.
        Returns a list of metadata dictionaries enriched with distance score.
        """
        if not self.index:
            raise ValueError("Index not loaded or built yet. Cannot search.")
            
        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                res = dict(self.metadata[idx])
                res["score"] = float(distances[0][i])
                results.append(res)
                
        return results

    def save(self, index_path: str = None, metadata_path: str = None):
        """
        Saves the FAISS index and the associated metadata map to disk.
        """
        out_index = index_path or self.index_path
        out_meta = metadata_path or self.metadata_path
        
        if not out_index or not out_meta:
            raise ValueError("Paths must be provided to save.")
            
        os.makedirs(os.path.dirname(out_index), exist_ok=True)
        faiss.write_index(self.index, out_index)
        
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Saved FAISS index to {out_index} and metadata to {out_meta}.")

    def load(self, index_path: str = None, metadata_path: str = None):
        """
        Loads the FAISS index and metadata from disk.
        """
        in_index = index_path or self.index_path
        in_meta = metadata_path or self.metadata_path
        
        self.index = faiss.read_index(in_index)
        with open(in_meta, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        logging.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
