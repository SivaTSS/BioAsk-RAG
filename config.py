# config.py

import os

# Paths to pre-built indexes
DENSE_INDEX_PATH = os.path.join('src', 'retrievers', 'dense_retriever_index.pkl')
SPARSE_INDEX_PATH = os.path.join('src', 'retrievers', 'sparse_retriever_index.pkl')

# Model names (using free models from HuggingFace)
LLM_MODEL_NAME = 'facebook/opt-350m'  # A free LLM alternative
TEXT_TO_SQL_MODEL = 'tscholak/langchain-sql'  # Example open-source Text-to-SQL model
TEXT_TO_CYPHER_MODEL = 'your-open-source-cypher-model'  # Placeholder, replace with actual model if available

# Database configurations
RELATIONAL_DB_URI = 'sqlite:///data/relational.db'  # Using SQLite for free relational DB
GRAPH_DB_URI = 'bolt://localhost:7687'  # Example for Neo4j (ensure Neo4j is installed and running)
VECTOR_DB_PATH = 'data/vector_db.faiss'

# Query Translation Techniques
QUERY_TRANSLATION_TECHNIQUES = [
    "Multi-query",
    "RAG-Fusion",
    "Decomposition",
    "Step-back",
    "HyDE"
]

# Routing prompt templates
SEMANTIC_PROMPT_TEMPLATES = [
    "Retrieve from Relational DB",
    "Retrieve from Graph DB",
    "Retrieve from Vector DB"
]
