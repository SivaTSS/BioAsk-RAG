# src/databases/graph_db.py

from neo4j import GraphDatabase
from langchain_community.llms import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import torch

class GraphDB:
    def __init__(self, uri=os.getenv('GRAPH_DB_URI', 'bolt://localhost:7687'), user='neo4j', password='password'):
        # Initialize the Neo4j driver
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Initialize open-source LLM using HuggingFace Pipeline for Text-to-Cypher
        model_name = os.getenv('TEXT_TO_CYPHER_MODEL', 'facebook/opt-350m')  # Replace with a suitable model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.llm = HuggingFacePipeline(
            pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        )

    def text_to_cypher(self, query):
        # Convert the natural language query to Cypher
        prompt = f"Translate the following question to a Cypher query for Neo4j:\nQuestion: {query}\nCypher Query:"
        cypher_query = self.llm(prompt, max_length=150)[0]['generated_text'].split("Cypher Query:")[-1].strip()
        return cypher_query

    def execute_query(self, cypher_query):
        # Run the Cypher query against the Neo4j database
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]

    def close(self):
        # Close the driver connection
        self.driver.close()
