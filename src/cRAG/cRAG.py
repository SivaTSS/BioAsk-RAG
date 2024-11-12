# src/crag/crag.py

from src.retrievers.retrieval_pipeline import RetrievalPipeline
from src.retrievers.crag import CRAG
from src.generation.active_retrieval_generation import ActiveRetrievalGeneration
from config import LLM_MODEL_NAME

class cRAG:
    def __init__(self, retrieval_method='RAG-Fusion'):
        self.retrieval_pipeline = RetrievalPipeline(method=retrieval_method)
        self.crag = CRAG()
        self.active_generation = ActiveRetrievalGeneration(retrieval_method)

    def get_compressed_relevant_docs(self, query):
        # Retrieve documents
        retrieved_docs = self.retrieval_pipeline.retrieve(query, top_k=5)
        # Compress and filter
        compressed_docs = self.crag.compress(retrieved_docs)
        return compressed_docs

    def answer_question(self, question):
        compressed_docs = self.get_compressed_relevant_docs(question)
        answer = self.active_generation.generate_answer(question, [compressed_docs])
        return answer
