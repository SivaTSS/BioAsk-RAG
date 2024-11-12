# src/retrievers/rerank.py

from langchain_community.llms import HuggingFacePipeline

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os
import torch

class ReRanker:
    def __init__(self):
        model_name = 'facebook/opt-350m'  # Replace with a suitable open-source model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.llm = HuggingFacePipeline(
            pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        )

    def rerank(self, query, documents):
        # Create a prompt that asks the model to rank documents based on relevance to the query
        prompt = f"Rank the following documents based on their relevance to the question:\n\nQuestion: {query}\n\nDocuments:\n"
        for idx, doc in enumerate(documents, 1):
            prompt += f"{idx}. {doc}\n"
        prompt += "\nProvide the ranked list of document numbers in order of relevance (most relevant first):"

        ranking = self.llm(prompt, max_length=50)[0]['generated_text']
        try:
            ranked_indices = [int(num.strip()) for num in ranking.split() if num.strip().isdigit()]
            ranked_docs = [documents[i-1] for i in ranked_indices if 0 < i <= len(documents)]
            return ranked_docs
        except:
            # In case of any parsing error, return the original list
            return documents
