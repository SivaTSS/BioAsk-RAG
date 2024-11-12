```markdown
# Comprehensive RAG Question Answering

## Overview
This project implements a **Question Answering** system using **Retrieval-Augmented Generation (RAG)**, combining multiple databases, query translation, routing strategies, and advanced retrieval and generation techniques to deliver precise, context-aware answers. Built on **LangChain** and open-source resources, it is cost-effective and accessible.

## Features

- **Query Construction**
  - **Relational DBs**: Converts questions to SQL via Text-to-SQL.
  - **GraphDBs**: Converts questions to Cypher using Text-to-Cypher.
  - **VectorDBs**: Uses a Self-query retriever with metadata filters.

- **Query Translation**
  - Techniques like **Multi-query**, **RAG-Fusion**, **Decomposition**, **Step-back**, and **HyDE** optimize query effectiveness.

- **Routing**
  - **Logical** and **Semantic Routing** direct queries based on content or embeddings.

- **Retrieval**
  - Techniques like **Re-Rank**, **RankGPT**, **CRAG**, and **Active Retrieval** enhance result relevance and filtering.

- **Indexing**
  - Optimizes document chunking, summaries, specialized embeddings, and hierarchical indexing.

- **Generation**
  - **Active Retrieval Generation**, **Self-RAG-RRR**, and **cRAG** methods improve generation quality.

- **Adaptive RAG**
  - Dynamically adjusts retrieval and generation strategies for optimal answers.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/comprehensive-rag-qa.git
   cd comprehensive-rag-qa
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Databases**
   - **Relational DB**: Ensure `relational.db` exists in the `data/` directory.
   - **Graph DB**: Start and populate Neo4j. Set the URI in `config.py`.
   - **Vector DB**: Create FAISS index and corpus files with `scripts/create_vector_db.py`.

5. **Configure Environment Variables**
   - Edit `config.py` to specify paths and model names.

## Usage

Start the application:
```bash
python main.py
```

Access the Gradio interface at the provided URL (usually http://127.0.0.1:7860).

### Interface Options
- **Question**: Enter your question.
- **Query Translation Technique**: Select a technique (e.g., Decomposition, RAG-Fusion).
- **Routing Strategy**: Choose Logical or Semantic.
- **Retrieval Method**: Pick a method (e.g., Retrieval Pipeline, CRAG, Adaptive RAG).
- **Generation Method**: Choose from Active Retrieval Generation, cRAG, etc.

## Configuration

Adjust settings in `config.py` for database URIs, model names, and other parameters.
