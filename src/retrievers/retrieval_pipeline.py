# src/retrievers/retrieval_pipeline.py

from .rerank import ReRanker
from .rankgpt import RankGPT
from .rag_fusion import RAGFusion
from .crag import CRAG
from .active_retrieval import ActiveRetrieval

class RetrievalPipeline:
    def __init__(self, method='RAG-Fusion'):
        self.reranker = ReRanker()
        self.rankgpt = RankGPT()
        self.rag_fusion = RAGFusion()
        self.crag = CRAG()
        self.active_retrieval = ActiveRetrieval()
        self.method = method

    def retrieve(self, query, top_k=5):
        # Initial retrieval using dense retriever
        initial_docs = self.dense_retriever.retrieve(query, top_k=top_k)
        
        # Apply ranking
        if self.method == 'Re-Rank':
            ranked_docs = self.reranker.rerank(query, initial_docs)
        elif self.method == 'RankGPT':
            ranked_docs = self.rankgpt.rank(query, initial_docs)
        elif self.method == 'RAG-Fusion':
            ranked_docs = self.rag_fusion.fusion_rank(query, initial_docs)
        else:
            ranked_docs = initial_docs
        
        # Apply refinement
        refined_docs = self.crag.compress(ranked_docs)
        # Optionally, apply active retrieval if needed
        final_docs = self.active_retrieval.retrieve(query, refined_docs)
        
        return final_docs
