# src/routing/semantic_routing.py

from sentence_transformers import SentenceTransformer, util
import numpy as np
from config import SEMANTIC_PROMPT_TEMPLATES

class SemanticRouter:
    def __init__(self, prompt_templates=SEMANTIC_PROMPT_TEMPLATES):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.prompt_templates = prompt_templates
        self.prompt_embeddings = self.model.encode(prompt_templates, convert_to_tensor=True)

    def route(self, question, top_k=1):
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(question_embedding, self.prompt_embeddings)[0]
        top_indices = similarities.topk(top_k).indices
        return [self.prompt_templates[idx] for idx in top_indices]
