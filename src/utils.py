# src/utils.py

import json
import pickle
from sentence_transformers import SentenceTransformer, util

def load_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def evaluate_relevance(question, documents):
    """
    Evaluates the relevance of each document to the question.
    Returns a list of similarity scores between 0 and 1.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    question_embedding = model.encode(question, convert_to_tensor=True)
    doc_embeddings = model.encode(documents, convert_to_tensor=True)
    cosine_scores = util.cos_sim(question_embedding, doc_embeddings)[0]
    relevance_scores = cosine_scores.cpu().numpy().tolist()
    # Normalize scores between 0 and 1
    max_score = max(relevance_scores) if relevance_scores else 1
    normalized_scores = [score / max_score if max_score > 0 else 0 for score in relevance_scores]
    return normalized_scores
