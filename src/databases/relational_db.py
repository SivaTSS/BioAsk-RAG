# src/databases/relational_db.py

from langchain_community.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline

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
        
        # Define the prompt for converting natural language to SQL
        self.prompt_template = PromptTemplate(
            input_variables=["question"],
            template="Translate the following question to SQL query:\nQuestion: {question}\nSQL Query:"
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def text_to_sql(self, query):
        # Run the question through the LLM chain to generate SQL
        sql_query = self.chain.run(question=query)
        
        # Execute the generated SQL query with SQLDatabase
        result = self.db.run(sql_query)
        return result
