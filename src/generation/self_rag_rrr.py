# src/generation/self_rag_rrr.py

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from src.retrievers.retrieval_pipeline import RetrievalPipeline
import os
import torch
from config import GENERATION_MODEL_NAME

class SelfRAG_RRR:
    def __init__(self, retrieval_method='RAG-Fusion'):
        self.retrieval_pipeline = RetrievalPipeline(method=retrieval_method)
        model_name = os.getenv('GENERATION_MODEL_NAME', GENERATION_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.llm = HuggingFacePipeline(
            pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        )

    def generate_answer(self, question):
        # Initial retrieval
        initial_docs = self.retrieval_pipeline.retrieve(question, top_k=5)
        
        # Generate initial answer
        context = "\n\n".join(initial_docs)
        prompt = f"Question: {question}\n\nContext: {context}\n\nAnswer:"
        initial_answer = self.llm(prompt, max_length=200)[0]['generated_text']
        
        # Evaluate answer quality (placeholder logic)
        if len(initial_answer.split()) < 50:  # Example threshold
            # Rewrite the question for better retrieval
            rewrite_prompt = f"Rewrite the following question to be more specific and improve retrieval quality:\n\nOriginal Question: {question}\n\nRewritten Question:"
            rewritten_question = self.llm(rewrite_prompt, max_length=100)[0]['generated_text'].split("Rewritten Question:")[-1].strip()
            
            # Re-retrieve with the rewritten question
            refined_docs = self.retrieval_pipeline.retrieve(rewritten_question, top_k=5)
            refined_context = "\n\n".join(refined_docs)
            refined_prompt = f"Question: {rewritten_question}\n\nContext: {refined_context}\n\nAnswer:"
            final_answer = self.llm(refined_prompt, max_length=200)[0]['generated_text']
            return final_answer
        return initial_answer
