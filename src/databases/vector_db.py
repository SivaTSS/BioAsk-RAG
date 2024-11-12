# src/databases/vector_db.py

import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, index_path=os.getenv('VECTOR_DB_PATH')):
        self.index = faiss.read_index(index_path)
        corpus_path = index_path.replace('.faiss', '_corpus.pkl')
        with open(corpus_path, 'rb') as f:
            self.corpus = pickle.load(f)
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def self_query_retrieve(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.corpus[idx] for idx in indices[0] if idx < len(self.corpus)]
