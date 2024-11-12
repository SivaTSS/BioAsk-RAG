# src/adaptive_RAG/adaptive_rag.py

from src.retrievers.retrieval_pipeline import RetrievalPipeline
from src.generation.active_retrieval_generation import ActiveRetrievalGeneration
from src.utils import evaluate_relevance
from config import SIMILARITY_THRESHOLD

class AdaptiveRAG:
    def __init__(self, retrieval_method='RAG-Fusion'):
        self.retrieval_pipeline = RetrievalPipeline(method=retrieval_method)
        self.active_generation = ActiveRetrievalGeneration(retrieval_method)

    def adapt_retrieval(self, question, initial_docs):
        # Evaluate relevance
        relevance_scores = evaluate_relevance(question, initial_docs)
        average_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        if average_relevance < SIMILARITY_THRESHOLD:  # Example threshold
            # Enhance retrieval parameters
            enhanced_docs = self.retrieval_pipeline.retrieve(question, top_k=10)
            return enhanced_docs
        return initial_docs

    def answer_question(self, question):
        initial_docs = self.retrieval_pipeline.retrieve(question, top_k=5)
        adapted_docs = self.adapt_retrieval(question, initial_docs)
        answer = self.active_generation.generate_answer(question, adapted_docs)
        return answer
