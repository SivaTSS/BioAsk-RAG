# src/databases/graph_db.py

from py2cypher import CypherSession
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import torch

class GraphDB:
    def __init__(self, uri=os.getenv('GRAPH_DB_URI'), user='neo4j', password='password'):
        self.session = CypherSession(uri, user, password)
        # Initialize open-source LLM using HuggingFace Pipeline for Text-to-Cypher
        model_name = os.getenv('TEXT_TO_CYPHER_MODEL', 'facebook/opt-350m')  # Replace with actual model if available
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.llm = HuggingFacePipeline(pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1))

    def text_to_cypher(self, query):
        prompt = f"Convert the following natural language query to Cypher query language for Neo4j:\n\nQuestion: {query}\nCypher Query:"
        cypher_query = self.llm(prompt, max_length=150)[0]['generated_text'].split("Cypher Query:")[-1].strip()
        return cypher_query

    def execute_query(self, cypher_query):
        return self.session.run(cypher_query)
