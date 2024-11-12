# src/retrievers/rag_fusion.py

from .rerank import ReRanker
from .rankgpt import RankGPT

class RAGFusion:
    def __init__(self):
        self.reranker = ReRanker()
        self.rankgpt = RankGPT()

    def fusion_rank(self, query, documents):
        # First, apply Re-Rank
        reranked_docs = self.reranker.rerank(query, documents)
        # Then, apply RankGPT on the reranked documents
        final_ranked_docs = self.rankgpt.rank(query, reranked_docs)
        return final_ranked_docs
