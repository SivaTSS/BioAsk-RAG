# main.py

import gradio as gr
from src.databases.relational_db import RelationalDB
from src.databases.graph_db import GraphDB
from src.databases.vector_db import VectorDB
from src.query_translation.translator import QueryTranslator
from src.routing.logical_routing import LogicalRouter
from src.routing.semantic_routing import SemanticRouter
from src.retrievers.retrieval_pipeline import RetrievalPipeline
from src.retrievers.crag import CRAG
from src.retrievers.rerank import ReRanker
from src.retrievers.rankgpt import RankGPT
from src.retrievers.rag_fusion import RAGFusion
from src.retrievers.active_retrieval import ActiveRetrieval
from src.indexing.indexing_pipeline import IndexingPipeline
from src.generation.active_retrieval_generation import ActiveRetrievalGeneration
from src.generation.self_rag_rrr import SelfRAG_RRR
from src.crag.crag import cRAG
from src.adaptive_RAG.adaptive_rag import AdaptiveRAG
from config import QUERY_TRANSLATION_TECHNIQUES, SEMANTIC_PROMPT_TEMPLATES

def load_components():
    # Initialize databases
    relational_db = RelationalDB()
    graph_db = GraphDB()
    vector_db = VectorDB()

    # Initialize query translator
    query_translator = QueryTranslator(technique="Decomposition")  # Default technique

    # Initialize routers
    logical_router = LogicalRouter()
    semantic_router = SemanticRouter(prompt_templates=SEMANTIC_PROMPT_TEMPLATES)

    # Initialize advanced components
    retrieval_pipeline = RetrievalPipeline(method='RAG-Fusion')
    crag = CRAG()
    active_generation = ActiveRetrievalGeneration(retrieval_method='RAG-Fusion')
    self_rag_rrr = SelfRAG_RRR(retrieval_method='RAG-Fusion')
    crag_system = cRAG(retrieval_method='RAG-Fusion')
    adaptive_rag = AdaptiveRAG(retrieval_method='RAG-Fusion')

    return {
        "relational_db": relational_db,
        "graph_db": graph_db,
        "vector_db": vector_db,
        "query_translator": query_translator,
        "logical_router": logical_router,
        "semantic_router": semantic_router,
        "retrieval_pipeline": retrieval_pipeline,
        "crag": crag,
        "active_generation": active_generation,
        "self_rag_rrr": self_rag_rrr,
        "crag_system": crag_system,
        "adaptive_rag": adaptive_rag
    }

def answer_question(question, translation_technique, routing_strategy, retrieval_method, generation_method):
    components = load_components()

    # Translate query
    components["query_translator"].technique = translation_technique
    translated_query = components["query_translator"].translate(question)

    # Route query
    if routing_strategy == "Logical Routing":
        db_choice = components["logical_router"].route(question)
    else:
        semantic_prompt = components["semantic_router"].route(question)[0]
        if "Relational" in semantic_prompt:
            db_choice = "RelationalDB"
        elif "Graph" in semantic_prompt:
            db_choice = "GraphDB"
        else:
            db_choice = "VectorDB"

    # Retrieval based on advanced methods
    if retrieval_method == "Retrieval Pipeline":
        retrieved_docs = components["retrieval_pipeline"].retrieve(translated_query, top_k=5)
    elif retrieval_method == "CRAG":
        retrieved_docs = components["crag_system"].get_compressed_relevant_docs(translated_query)
    elif retrieval_method == "Adaptive RAG":
        retrieved_docs = components["adaptive_rag"].answer_question(translated_query)
    elif retrieval_method == "Self-RAG-RRR":
        retrieved_docs = components["self_rag_rrr"].answer_question(translated_query)
    else:
        retrieved_docs = []

    # Generation based on advanced methods
    if generation_method == "Active Retrieval Generation":
        answer = components["active_generation"].generate_answer(question, retrieved_docs)
    elif generation_method == "Self-RAG-RRR":
        answer = components["self_rag_rrr"].generate_answer(question)
    elif generation_method == "cRAG":
        answer = components["crag_system"].answer_question(question)
    elif generation_method == "Adaptive RAG":
        answer = components["adaptive_rag"].answer_question(question)
    else:
        # Default generation
        answer = "Generation method not selected."

    return answer

# Define Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.components.Textbox(lines=2, placeholder="Enter your question here...", label="Question"),
        gr.components.Dropdown(
            choices=QUERY_TRANSLATION_TECHNIQUES,
            label="Select Query Translation Technique",
            value="Decomposition"
        ),
        gr.components.Radio(
            choices=["Logical Routing", "Semantic Routing"],
            label="Select Routing Strategy",
            value="Logical Routing"
        ),
        gr.components.Dropdown(
            choices=["Retrieval Pipeline", "CRAG", "Adaptive RAG", "Self-RAG-RRR"],
            label="Select Retrieval Method",
            value="Retrieval Pipeline"
        ),
        gr.components.Dropdown(
            choices=["Active Retrieval Generation", "Self-RAG-RRR", "cRAG", "Adaptive RAG"],
            label="Select Generation Method",
            value="Active Retrieval Generation"
        )
    ],
    outputs=gr.components.Textbox(label="Answer"),
    title="Comprehensive RAG Question Answering",
    description="A sophisticated Question Answering system leveraging Retrieval-Augmented Generation (RAG) with advanced Retrieval, Indexing, and Generation techniques. Select various methods to explore different configurations and their performance.",
    theme="default",
    examples=[
        ["What are the records in the Sales table?", "Decomposition", "Logical Routing", "Retrieval Pipeline", "Active Retrieval Generation"],
        ["Show the relationships between users and products.", "RAG-Fusion", "Semantic Routing", "CRAG", "Self-RAG-RRR"],
        ["Retrieve documents related to machine learning.", "Multi-query", "Semantic Routing", "Adaptive RAG", "cRAG"],
        # Add more examples as needed
    ],
    allow_flagging="never",
    css="""
    .gradio-container {
        background-color: #f0f2f5;
        padding: 30px;
    }
    .gr-button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
"""
)

if __name__ == "__main__":
    iface.launch()
