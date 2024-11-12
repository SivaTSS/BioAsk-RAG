# src/retrievers/rankgpt.py

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import os
import torch
from config import LLM_MODEL_NAME

class RankGPT:
    def __init__(self):
        model_name = os.getenv('LLM_MODEL_NAME', LLM_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.llm = HuggingFacePipeline(
            pipeline('text2text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        )

    def rank(self, query, documents):
        # Similar to Re-Rank but can include more sophisticated prompts or logic
        prompt = f"Given the question: {query}\n\nRank the following documents from most relevant to least relevant:\n"
        for idx, doc in enumerate(documents, 1):
            prompt += f"{idx}. {doc}\n"
        prompt += "\nProvide the ranked list of document numbers in order of relevance:"

        ranking = self.llm(prompt, max_length=50)[0]['generated_text']
        try:
            ranked_indices = [int(num.strip()) for num in ranking.split() if num.strip().isdigit()]
            ranked_docs = [documents[i-1] for i in ranked_indices if 0 < i <= len(documents)]
            return ranked_docs
        except:
            return documents
