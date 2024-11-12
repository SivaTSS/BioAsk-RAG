# src/query_translation/translator.py

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import torch
from config import LLM_MODEL_NAME

class QueryTranslator:
    def __init__(self, technique="Decomposition"):
        self.technique = technique
        model_name = os.getenv('LLM_MODEL_NAME', LLM_MODEL_NAME)  # Use config
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.llm = HuggingFacePipeline(
            pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        )
        self.prompt_templates = {
            "Decomposition": "Break down the following question into simpler sub-questions:\n\nQuestion: {question}\nSub-questions:",
            "RAG-Fusion": "Translate the following question into a form optimized for retrieval:\n\nQuestion: {question}\nTranslated Query:",
            "Multi-query": "Generate multiple queries for retrieving information related to the following question:\n\nQuestion: {question}\nQueries:",
            "Step-back": "Rephrase the following question to enhance its clarity and retrieval effectiveness:\n\nQuestion: {question}\nRephrased Question:",
            "HyDE": "Generate hypothetical context to answer the following question:\n\nQuestion: {question}\nHypothetical Context:"
        }

    def translate(self, question):
        if self.technique not in self.prompt_templates:
            raise ValueError(f"Unsupported translation technique: {self.technique}")
        prompt = self.prompt_templates[self.technique].format(question=question)
        translated_output = self.llm(prompt, max_length=150)[0]['generated_text']
        # Extract the translated part after the prompt's label
        if self.technique == "Decomposition":
            return translated_output.split("Sub-questions:")[-1].strip().split('\n')
        elif self.technique == "RAG-Fusion":
            return translated_output.split("Translated Query:")[-1].strip()
        elif self.technique == "Multi-query":
            return translated_output.split("Queries:")[-1].strip().split('\n')
        elif self.technique == "Step-back":
            return translated_output.split("Rephrased Question:")[-1].strip()
        elif self.technique == "HyDE":
            return translated_output.split("Hypothetical Context:")[-1].strip()
        else:
            return translated_output.strip()
