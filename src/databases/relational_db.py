# src/databases/relational_db.py

from langchain import SQLDatabase, SQLDatabaseChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import torch

class RelationalDB:
    def __init__(self, db_uri=os.getenv('RELATIONAL_DB_URI')):
        self.db = SQLDatabase.from_uri(db_uri)
        # Initialize open-source LLM using HuggingFace Pipeline
        model_name = os.getenv('TEXT_TO_SQL_MODEL', 'tscholak/langchain-sql')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.llm = HuggingFacePipeline(pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1))
        self.chain = SQLDatabaseChain.from_llm(self.llm, self.db, verbose=True)

    def text_to_sql(self, query):
        return self.chain.run(query)
