from sentence_transformers import SentenceTransformer, util
import pickle
from config import DENSE_INDEX_PATH, DENSE_RETRIEVER_MODEL
from src.utils import load_pickle

class DenseRetriever:
    def __init__(self, model_name=DENSE_RETRIEVER_MODEL, index_path=DENSE_INDEX_PATH):
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.corpus_embeddings = None
        self.load_index(index_path)

    def load_index(self, index_path):
        data = load_pickle(index_path)
        self.corpus = data['corpus']
        self.corpus_embeddings = data['embeddings']
    
    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)
        return [self.corpus[hit['corpus_id']] for hit in hits[0]]
    
    # Implement additional retrieval methods as needed
    def retrieve_with_graph(self, query, top_k=5):
        # Placeholder for graph-based retrieval logic
        return self.retrieve(query, top_k)
    
    def retrieve_self_supervised(self, query, top_k=5):
        # Placeholder for self-supervised retrieval logic
        return self.retrieve(query, top_k)
    
    def retrieve_efficiently(self, query, top_k=5):
        # Placeholder for efficient retrieval logic
        return self.retrieve(query, top_k)