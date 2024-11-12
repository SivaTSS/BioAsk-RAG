# src/indexing/indexing_pipeline.py

from .semantic_splitter import SemanticSplitter
from .multi_representation_indexing import MultiRepresentationIndexer
from .dense_x import DenseX
from .specialized_embeddings import SpecializedEmbeddings
from .raptor import RAPTOR

class IndexingPipeline:
    def __init__(self):
        self.splitter = SemanticSplitter()
        self.multi_rep_indexer = MultiRepresentationIndexer()
        self.dense_x = DenseX()
        self.specialized_embeddings = SpecializedEmbeddings()
        self.raptor = RAPTOR()

    def process_documents(self, documents):
        # Step 1: Chunk Optimization
        chunks = []
        for doc in documents:
            chunks.extend(self.splitter.split(doc))
        
        # Step 2: Multi-representation Indexing (Summarization)
        summaries = [self.multi_rep_indexer.summarize(chunk) for chunk in chunks]
        
        # Step 3: Dense X Conversion
        dense_reps = self.dense_x.convert(summaries)
        
        # Step 4: Specialized Embeddings
        embeddings = self.specialized_embeddings.embed(dense_reps)
        
        # Step 5: Hierarchical Indexing (RAPTOR)
        hierarchical_summary = self.raptor.hierarchical_summarize(dense_reps)
        
        return {
            "chunks": chunks,
            "summaries": summaries,
            "dense_reps": dense_reps,
            "embeddings": embeddings,
            "hierarchical_summary": hierarchical_summary
        }
