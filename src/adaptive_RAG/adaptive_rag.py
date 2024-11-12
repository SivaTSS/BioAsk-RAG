# src/adaptive_RAG/adaptive_rag.py

from .retrieval_pipeline import RetrievalPipeline
from .generation.active_retrieval_generation import ActiveRetrievalGeneration
from .utils import evaluate_relevance  # Placeholder for relevance evaluation function

class AdaptiveRAG:
    def __init__(self, retrieval_method='RAG-Fusion'):
        self.retrieval_pipeline = RetrievalPipeline(method=retrieval_method)
        self.active_generation = ActiveRetrievalGeneration(retrieval_method)

    def adapt_retrieval(self, question, initial_docs):
        # Placeholder logic for adapting retrieval
        relevance_scores = evaluate_relevance(question, initial_docs)
        if sum(relevance_scores) / len(relevance_scores) < 0.7:  # Example threshold
            # Enhance retrieval parameters
            enhanced_docs = self.retrieval_pipeline.retrieve(question, top_k=10)
            return enhanced_docs
        return initial_docs

    def answer_question(self, question):
        initial_docs = self.retrieval_pipeline.retrieve(question, top_k=5)
        adapted_docs = self.adapt_retrieval(question, initial_docs)
        answer = self.active_generation.generate_answer(question, adapted_docs)
        return answer
