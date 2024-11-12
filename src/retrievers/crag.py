# src/retrievers/crag.py

from langchain_community.llms import HuggingFacePipeline

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os
import torch

class CRAG:
    def __init__(self):
        model_name = 'facebook/opt-350m'  # Replace with a suitable open-source model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.llm = HuggingFacePipeline(
            pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        )

    def compress(self, documents):
        # Combine documents and ask the model to compress them
        combined_docs = "\n\n".join(documents)
        prompt = f"Compress the following documents into a concise summary while retaining all key information:\n\n{combined_docs}\n\nSummary:"
        summary = self.llm(prompt, max_length=500)[0]['generated_text']
        return summary
