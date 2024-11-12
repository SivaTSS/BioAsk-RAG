# src/generation/active_retrieval_generation.py

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from src.retrievers.retrieval_pipeline import RetrievalPipeline
import os
import torch
from config import GENERATION_MODEL_NAME

class ActiveRetrievalGeneration:
    def __init__(self, retrieval_method='RAG-Fusion'):
        self.retrieval_pipeline = RetrievalPipeline(method=retrieval_method)
        model_name = os.getenv('GENERATION_MODEL_NAME', GENERATION_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.llm = HuggingFacePipeline(
            pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        )

    def generate_answer(self, question, initial_docs):
        # Generate an initial answer
        context = "\n\n".join(initial_docs)
        prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
        initial_answer = self.llm(prompt, max_length=200)[0]['generated_text']
        
        # Assess answer quality (placeholder logic)
        if len(initial_answer.split()) < 50:  # Example threshold
            # Re-retrieve documents
            refined_docs = self.retrieval_pipeline.retrieve(question, top_k=5)
            refined_context = "\n\n".join(refined_docs)
            prompt = f"Question: {question}\n\nRefined Context: {refined_context}\n\nAnswer:"
            final_answer = self.llm(prompt, max_length=200)[0]['generated_text']
            return final_answer
        return initial_answer
