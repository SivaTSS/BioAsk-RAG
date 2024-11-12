# src/indexing/specialized_embeddings.py

from sentence_transformers import SentenceTransformer

class SpecializedEmbeddings:
    def __init__(self, model_name='bert-base-nli-mean-tokens'):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)
