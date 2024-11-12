# src/indexing/raptor.py

from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os
import torch

class RAPTOR:
    def __init__(self):
        model_name = 'facebook/opt-350m'  # Replace with a suitable open-source model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.llm = HuggingFacePipeline(
            pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        )

    def hierarchical_summarize(self, documents, levels=3):
        current_docs = documents
        for level in range(levels):
            prompt = f"Summarize the following documents at level {level + 1} abstraction:\n\n" + "\n\n".join(current_docs) + "\n\nSummary:"
            summary = self.llm(prompt, max_length=300)[0]['generated_text']
            current_docs = [summary]
        return current_docs[0]
