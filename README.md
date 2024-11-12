### **Part 2: Comprehensive Documentation (`DOC.md`)**

Below is the comprehensive documentation for the **Comprehensive RAG Question Answering** system. This documentation covers all the specified topics, detailing each module, its purpose, functionalities, and interactions within the system.

---

# Comprehensive RAG Question Answering Documentation

## Table of Contents

1. [Overview](#overview)
2. [Modules and Concepts](#modules-and-concepts)
    - [1. Query Construction](#1-query-construction)
        - [1.1 Relational Databases (`RelationalDB`)](#11-relational-databases-relationaldb)
        - [1.2 Text-to-SQL](#12-text-to-sql)
        - [1.3 Graph Databases (`GraphDB`)](#13-graph-databases-graphdb)
        - [1.4 Text-to-Cypher](#14-text-to-cypher)
        - [1.5 Vector Databases (`VectorDB`)](#15-vector-databases-vectordb)
        - [1.6 Self-Query Retriever](#16-self-query-retriever)
    - [2. Query Translation](#2-query-translation)
        - [2.1 Input Question Translation](#21-input-question-translation)
        - [2.2 Translation Techniques](#22-translation-techniques)
    - [3. Routing](#3-routing)
        - [3.1 Logical Routing (`LogicalRouter`)](#31-logical-routing-logicalrouter)
        - [3.2 Semantic Routing (`SemanticRouter`)](#32-semantic-routing-semanticrouter)
    - [4. Retrieval](#4-retrieval)
        - [4.1 Ranking](#41-ranking)
        - [4.2 Refinement](#42-refinement)
    - [5. Indexing](#5-indexing)
        - [5.1 Chunk Optimization (`SemanticSplitter`)](#51-chunk-optimization-semanticsplitter)
        - [5.2 Multi-representation Indexing (`MultiRepresentationIndexer`)](#52-multi-representation-indexing-multirepresentationindexer)
        - [5.3 Specialized Embeddings (`SpecializedEmbeddings`)](#53-specialized-embeddings-specializedembeddings)
        - [5.4 Hierarchical Indexing (`RAPTOR`)](#54-hierarchical-indexing-raptor)
    - [6. Generation](#6-generation)
        - [6.1 Active Retrieval](#61-active-retrieval)
    - [7. Adaptive RAG (`AdaptiveRAG`)](#7-adaptive-rag-adaptiverag)
3. [Configuration (`config.py`)](#configuration-configpy)
4. [Usage](#usage)
5. [Best Practices](#best-practices)
6. [Additional Information](#additional-information)

---

## Overview

The **Comprehensive RAG Question Answering** system is a sophisticated framework designed to answer user queries by leveraging **Retrieval-Augmented Generation (RAG)** techniques. It integrates multiple databases, advanced query translation methods, intelligent routing strategies, and state-of-the-art retrieval and generation mechanisms to deliver precise and context-aware responses.

Key Features:

- **Multi-Database Support**: Handles relational databases (SQL), graph databases (Cypher), and vector databases.
- **Advanced Query Translation**: Transforms natural language questions into optimized queries using various techniques.
- **Intelligent Routing**: Determines the most appropriate database and retrieval method based on the query's semantics.
- **Enhanced Retrieval**: Utilizes ranking, re-ranking, and fusion methods to fetch the most relevant documents.
- **Optimized Indexing**: Implements chunk optimization, multi-representation indexing, specialized embeddings, and hierarchical summarization.
- **Dynamic Generation**: Employs active retrieval and self-refinement strategies to generate high-quality answers.
- **Adaptive RAG**: Adjusts retrieval and generation strategies dynamically based on data relevance and quality.

---

### Key Directories and Files

- **`config.py`**: Centralized configuration file containing all model names, database URIs, and other configurable parameters.
- **`src/`**: Contains all source code modules, organized into subdirectories based on functionality.

---

## Modules and Concepts

### 1. Query Construction

**Purpose**: Converts user-provided natural language questions into structured queries suitable for different types of databases (Relational, Graph, Vector).

#### 1.1 Relational Databases (`RelationalDB`)

**Location**: `src/databases/relational_db.py`

**Description**: Manages interactions with relational databases using SQL. Translates natural language queries into SQL statements.

**Key Components**:
- **`text_to_sql(query)`**: Transforms a natural language question into an SQL query using a Text-to-SQL model.
- **`execute_query(sql_query)`**: Executes the generated SQL query against the relational database and retrieves results.

#### 1.2 Text-to-SQL

**Description**: Utilizes a Language Model to convert natural language questions into SQL queries.

**Implementation**:
- Integrated within the `RelationalDB` module.
- Uses HuggingFace's `HuggingFacePipeline` with a specified Text-to-SQL model (`tscholak/langchain-sql` by default).

#### 1.3 Graph Databases (`GraphDB`)

**Location**: `src/databases/graph_db.py`

**Description**: Manages interactions with graph databases (e.g., Neo4j) using the Cypher query language. Translates natural language questions into Cypher queries.

**Key Components**:
- **`text_to_cypher(query)`**: Converts a natural language question into a Cypher query using a Text-to-Cypher model.
- **`execute_query(cypher_query)`**: Executes the Cypher query against the graph database and retrieves results.

#### 1.4 Text-to-Cypher

**Description**: Utilizes a Language Model to convert natural language questions into Cypher queries.

**Implementation**:
- Integrated within the `GraphDB` module.
- Uses HuggingFace's `HuggingFacePipeline` with a specified Text-to-Cypher model (`your-open-source-cypher-model` placeholder).

#### 1.5 Vector Databases (`VectorDB`)

**Location**: `src/databases/vector_db.py`

**Description**: Manages interactions with vector databases using FAISS for efficient similarity search. Retrieves documents based on vector embeddings.

**Key Components**:
- **`self_query_retrieve(query, top_k=5)`**: Retrieves the top `k` most similar documents to the query based on vector embeddings.

#### 1.6 Self-Query Retriever

**Description**: Automatically generates metadata filters from query data to enhance retrieval performance in vector databases.

**Implementation**:
- Integrated within the `VectorDB` module.
- Utilizes embeddings from `SentenceTransformer` models for similarity computations.

---

### 2. Query Translation

**Purpose**: Transforms user questions into optimized queries suitable for effective retrieval.

#### 2.1 Input Question Translation

**Description**: Translates the user's natural language question into a form that enhances the retrieval process, making it more precise and relevant.

**Implementation**:
- Managed by the `QueryTranslator` module (`src/query_translation/translator.py`).
- Supports multiple translation techniques as defined in the configuration.

#### 2.2 Translation Techniques

**Description**: Implements various methods to optimize query translation, enhancing the effectiveness of information retrieval.

**Supported Techniques**:

1. **Multi-query**: Generates multiple related queries to fetch a broader set of relevant documents.
2. **RAG-Fusion**: Translates the question into a form that integrates Retrieval-Augmented Generation for more accurate results.
3. **Decomposition**: Breaks down complex questions into simpler sub-questions to improve retrieval precision.
4. **Step-back**: Rephrases questions to enhance clarity and retrieval effectiveness.
5. **HyDE**: Generates hypothetical contexts to aid in answering the question more effectively.

**Implementation**:
- Handled within the `QueryTranslator` class.
- Each technique has a corresponding prompt template to guide the translation process.

---

### 3. Routing

**Purpose**: Determines the most appropriate database and retrieval strategy based on the semantics of the user's question.

#### 3.1 Logical Routing (`LogicalRouter`)

**Location**: `src/routing/logical_routing.py`

**Description**: Routes queries to the appropriate database (RelationalDB, GraphDB, VectorDB) based on keyword matching within the question.

**Key Components**:
- **`route(question)`**: Analyzes the question for specific keywords to decide the target database.

**Implementation**:
- Utilizes predefined keyword sets for each database type to match against the user's question.

#### 3.2 Semantic Routing (`SemanticRouter`)

**Location**: `src/routing/semantic_routing.py`

**Description**: Routes queries based on semantic similarity between the question and predefined prompt templates, ensuring a more nuanced and context-aware routing decision.

**Key Components**:
- **`route(question, top_k=1)`**: Computes the semantic similarity between the question and prompt templates, returning the most similar prompts.

**Implementation**:
- Uses `SentenceTransformer` models to generate embeddings and compute cosine similarities.
- Selects the top `k` prompt templates that best match the question's semantics.

---

### 4. Retrieval

**Purpose**: Fetches relevant documents or data segments from the selected database to support answer generation.

#### 4.1 Ranking

**Description**: Orders retrieved documents based on their relevance to the user's query to ensure that the most pertinent information is prioritized.

**Supported Techniques**:

1. **Re-Rank**: Reorders documents based on additional relevance criteria using language models.
2. **RankGPT**: Utilizes GPT-based models to rank documents by relevance.
3. **RAG-Fusion**: Combines multiple ranking strategies to optimize document relevance.

**Implementation**:
- Managed by the `RetrievalPipeline` module (`src/retrievers/retrieval_pipeline.py`).
- Integrates components like `ReRanker`, `RankGPT`, and `RAGFusion` to apply selected ranking methods.

#### 4.2 Refinement

**Description**: Enhances the set of retrieved documents by filtering out irrelevant information and compressing content to focus on key details.

**Supported Techniques**:

1. **CRAG**: Compresses and filters relevant documents to streamline the information.
2. **Active Retrieval**: Dynamically retrieves additional documents if initial results are insufficient.

**Implementation**:
- Handled within the `RetrievalPipeline` through the `CRAG` and `ActiveRetrieval` components.

---

### 5. Indexing

**Purpose**: Preprocesses and organizes documents to facilitate efficient and effective retrieval operations.

#### 5.1 Chunk Optimization (`SemanticSplitter`)

**Location**: `src/indexing/semantic_splitter.py`

**Description**: Splits documents into optimized chunks based on semantic boundaries to improve embedding quality and retrieval accuracy.

**Key Components**:
- **`split(text)`**: Divides text into chunks with specified size and overlap parameters.

**Implementation**:
- Utilizes `RecursiveCharacterTextSplitter` from `langchain` to perform semantic-based splitting.

#### 5.2 Multi-representation Indexing (`MultiRepresentationIndexer`)

**Location**: `src/indexing/multi_representation_indexing.py`

**Description**: Generates multiple representations (e.g., summaries) of document chunks to support diverse retrieval and generation needs.

**Key Components**:
- **`summarize(document)`**: Produces concise summaries of document chunks while retaining key information.

**Implementation**:
- Employs a Language Model via HuggingFace's `HuggingFacePipeline` to generate summaries.

#### 5.3 Specialized Embeddings (`SpecializedEmbeddings`)

**Location**: `src/indexing/specialized_embeddings.py`

**Description**: Creates domain-specific or advanced embeddings for document chunks to enhance semantic understanding and retrieval performance.

**Key Components**:
- **`embed(texts)`**: Generates embeddings for a list of texts using specialized models.

**Implementation**:
- Uses `SentenceTransformer` models (e.g., `bert-base-nli-mean-tokens`) to compute embeddings.

#### 5.4 Hierarchical Indexing (`RAPTOR`)

**Location**: `src/indexing/raptor.py`

**Description**: Implements a hierarchical summarization approach, condensing documents at multiple abstraction levels to facilitate efficient retrieval.

**Key Components**:
- **`hierarchical_summarize(documents, levels=3)`**: Generates summaries at various abstraction levels.

**Implementation**:
- Uses a tree-based summarization process with a Language Model to iteratively condense information.

---

### 6. Generation

**Purpose**: Produces coherent and contextually relevant answers based on the retrieved documents.

#### 6.1 Active Retrieval (`ActiveRetrievalGeneration`)

**Location**: `src/generation/active_retrieval_generation.py`

**Description**: Generates answers by actively retrieving additional documents if initial retrievals are insufficient, ensuring high-quality responses.

**Key Components**:
- **`generate_answer(question, initial_docs)`**: Creates an answer using the provided documents. If the initial answer is deemed low quality, it triggers re-retrieval for refinement.

**Implementation**:
- Utilizes a Language Model to generate answers based on the context from retrieved documents.
- Integrates with the `RetrievalPipeline` to fetch additional documents when necessary.

---

### 7. Adaptive RAG (`AdaptiveRAG`)

**Location**: `src/adaptive_RAG/adaptive_rag.py`

**Description**: Dynamically adjusts retrieval and generation strategies based on the quality and relevance of the retrieved data, optimizing the RAG process for each query.

**Key Components**:
- **`adapt_retrieval(question, initial_docs)`**: Evaluates the relevance of initial documents and adjusts retrieval parameters if necessary.
- **`answer_question(question)`**: Manages the end-to-end process of retrieving, adapting, and generating answers.

**Implementation**:
- Uses relevance evaluation functions to assess the adequacy of retrieved documents.
- Enhances retrieval by fetching more documents or adjusting retrieval strategies based on predefined thresholds.

---

## Configuration (`config.py`)

All configurable parameters, such as model names, database URIs, paths, and thresholds, are centralized in `config.py`. This promotes maintainability and flexibility, allowing easy adjustments without modifying the core codebase.

### Key Configurations:

- **Model Names**:
    - `LLM_MODEL_NAME`: Name of the Language Model used across various modules.
    - `TEXT_TO_SQL_MODEL`: Model used for Text-to-SQL translation.
    - `TEXT_TO_CYPHER_MODEL`: Model used for Text-to-Cypher translation.

- **Database URIs and Paths**:
    - `RELATIONAL_DB_URI`: URI for the relational database (e.g., SQLite).
    - `GRAPH_DB_URI`: URI for the graph database (e.g., Neo4j).
    - `VECTOR_DB_PATH`: Path to the vector database index (FAISS).

- **Techniques and Templates**:
    - `QUERY_TRANSLATION_TECHNIQUES`: List of supported query translation techniques.
    - `SEMANTIC_PROMPT_TEMPLATES`: Predefined prompts for semantic routing.

- **Model and Indexing Parameters**:
    - `DENSE_RETRIEVER_MODEL`: Model used for dense retrieval.
    - `SPECIALIZED_EMBEDDING_MODEL`: Model used for specialized embeddings.
    - `RAPTOR_LEVELS`: Number of abstraction levels in hierarchical indexing.

- **Thresholds**:
    - `SIMILARITY_THRESHOLD`: Threshold for determining the relevance of retrieved documents.


## Usage

### **Setup Instructions**

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

    - Edit `config.py` to specify paths and model names as needed.

6. **Run the Application**

    ```bash
    python main.py
    ```

7. **Access the Interface**

    - Open the provided URL (usually [http://127.0.0.1:7860](http://127.0.0.1:7860)) in your web browser to interact with the Gradio interface.

### **Using the Gradio Interface**

- **Question**: Enter your natural language question.
- **Query Translation Technique**: Select from available techniques (e.g., Decomposition, RAG-Fusion).
- **Routing Strategy**: Choose between Logical or Semantic Routing.
- **Retrieval Method**: Select the retrieval strategy (e.g., Retrieval Pipeline, CRAG).
- **Generation Method**: Pick the generation approach (e.g., Active Retrieval Generation).

**Examples**:

1. **Relational Query**:
    - **Question**: "What are the records in the Sales table?"
    - **Translation Technique**: Decomposition
    - **Routing Strategy**: Logical Routing
    - **Retrieval Method**: Retrieval Pipeline
    - **Generation Method**: Active Retrieval Generation

2. **Graph Query**:
    - **Question**: "Show the relationships between users and products."
    - **Translation Technique**: RAG-Fusion
    - **Routing Strategy**: Semantic Routing
    - **Retrieval Method**: CRAG
    - **Generation Method**: Self-RAG-RRR

3. **Vector Retrieval**:
    - **Question**: "Retrieve documents related to machine learning."
    - **Translation Technique**: Multi-query
    - **Routing Strategy**: Semantic Routing
    - **Retrieval Method**: Adaptive RAG
    - **Generation Method**: cRAG

---

## Additional Information

### **Dependencies**

- **Transformers**: For language model implementations.
- **LangChain**: Facilitates the chaining of language models with various utilities.
- **Gradio**: Provides a user-friendly web interface for interacting with the system.
- **SentenceTransformers**: Used for generating and managing embeddings.
- **FAISS**: Efficient similarity search library for vector databases.
- **SQLAlchemy**: ORM for interacting with relational databases.
- **Neo4j**: Graph database management system.
