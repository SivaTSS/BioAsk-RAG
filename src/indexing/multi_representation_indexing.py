# src/indexing/multi_representation_indexing.py

from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os
import torch

class MultiRepresentationIndexer:
    def __init__(self):
        model_name = 'facebook/opt-350m'  # Replace with a suitable open-source model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.llm = HuggingFacePipeline(
            pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        )

    def summarize(self, document):
        prompt = f"Summarize the following document concisely while retaining all key information:\n\n{document}\n\nSummary:"
        summary = self.llm(prompt, max_length=150)[0]['generated_text']
        return summary
