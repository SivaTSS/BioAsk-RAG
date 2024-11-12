# Advanced RAG Question Answering

## Overview

This project implements a **Question Answering** system using **Retrieval-Augmented Generation (RAG)**. It integrates multiple databases, query translation techniques, and routing strategies to provide accurate and contextually relevant answers. The system leverages **LangChain** and entirely free resources to ensure accessibility and cost-effectiveness.

## Features

1. **Query Construction**
    - **Relational DBs**: Converts natural language questions to SQL queries using Text-to-SQL.
    - **GraphDBs**: Converts natural language questions to Cypher queries using Text-to-Cypher.
    - **VectorDBs**: Utilizes vector-based retrieval with a Self-query retriever to auto-generate metadata filters from query data.

2. **Query Translation**
    - Supports various techniques like **Multi-query**, **RAG-Fusion**, **Decomposition**, **Step-back**, and **HyDE** to optimize queries for retrieval.

3. **Routing**
    - **Logical Routing**: Routes queries based on predefined logical rules derived from question content.
    - **Semantic Routing**: Routes queries based on semantic similarity using embeddings and predefined prompt templates.

## Installation

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/advanced-rag-qa.git
cd advanced-rag-qa
