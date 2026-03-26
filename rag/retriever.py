import os
import json
import logging
import numpy as np

from config import PROCESSED_DIR, VECTOR_STORE_DIR
from embeddings.embedder import Embedder
from embeddings.vector_store import FAISSVectorStore

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "scraped_docs.index")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "scraped_metadata.json")

def build_retriever_index():
    """
    Reads the processed chunks, generates embeddings, and builds/saves the FAISS index.
    """
    chunks_path = os.path.join(PROCESSED_DIR, "chunks.json")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}. Run Stage 1 scraping first.")
        
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    logging.info(f"Loaded {len(chunks)} chunks for indexing.")
    
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings
    embedder = Embedder()
    embeddings_list = embedder.embed_texts(texts)
    embeddings_np = np.array(embeddings_list).astype('float32')
    
    # Build and save FAISS index
    store = FAISSVectorStore()
    store.build_index(embeddings_np, chunks)
    store.save(INDEX_PATH, METADATA_PATH)
    
    return store

class Retriever:
    """
    Provides a high-level interface to retrieve context for a query.
    """
    def __init__(self):
        self.embedder = Embedder()
        
        # Load index or build it if it doesn't exist
        if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
            logging.info("FAISS index not found. Building it now...")
            self.store = build_retriever_index()
        else:
            self.store = FAISSVectorStore(INDEX_PATH, METADATA_PATH)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieves the top_k most relevant chunks for the given query.
        """
        query_emb = self.embedder.embed_query(query)
        query_emb_np = np.array(query_emb).astype('float32')
        
        results = self.store.search(query_emb_np, top_k=top_k)
        return results

    def get_context_string(self, query: str, top_k: int = 3) -> str:
        """
        Retrieves top_k chunks and formats them into a single context string for an LLM prompt.
        """
        results = self.retrieve(query, top_k=top_k)
        
        context_parts = []
        for i, res in enumerate(results):
            title = res.get("title", "Unknown Source")
            url = res.get("url", "No URL")
            text = res.get("text", "")
            context_parts.append(f"[Source {i+1}: {title} ({url})]\n{text}")
            
        return "\n\n".join(context_parts)

if __name__ == "__main__":
    # Test script to build the index and retrieve a query
    logging.info("Initializing Retriever (will build index if necessary)...")
    retriever = Retriever()
    
    test_query = "What telematics products does Transight offer?"
    logging.info(f"Testing retrieval for query: '{test_query}'")
    
    context = retriever.get_context_string(test_query, top_k=3)
    print("\n--- RETRIEVED CONTEXT ---")
    print(context)
    print("-------------------------\n")
