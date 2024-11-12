# src/retrievers/active_retrieval.py

from .dense_retriever import DenseRetriever
from src.databases.vector_db import VectorDB

class ActiveRetrieval:
    def __init__(self):
        self.dense_retriever = DenseRetriever()
        self.vector_db = VectorDB()

    def retrieve(self, query, initial_docs, threshold=0.5):
        # Placeholder for relevance checking logic
        # For simplicity, assume if initial_docs are insufficient, retrieve again
        if not initial_docs or len(initial_docs) < threshold * 5:
            # Re-retrieve from the vector DB or other sources
            new_docs = self.vector_db.self_query_retrieve(query, top_k=5)
            return new_docs
        return initial_docs
