# Install required packages if not already installed:
# pip install langchain langchain-community langchain-huggingface transformers sentence-transformers faiss-cpu

import json
import os
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_bioasq_dataset(file_path):
    """Load the BioASQ dataset from a JSON file and return documents and QA pairs."""
    print("Loading BioASQ dataset from:", file_path)
    with open(file_path, 'r') as f:
        data = json.load(f)
    documents = []
    qa_pairs = []
    for entry in data['data']:
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            # Each paragraph becomes a Document
            documents.append(Document(page_content=context))
            for qa in paragraph['qas']:
                question = qa['question']
                answers = [a['text'] for a in qa['answers']]
                qa_pairs.append({'question': question, 'context': context, 'answers': answers})
    print(f"Loaded {len(documents)} documents and {len(qa_pairs)} QA pairs.")
    return documents, qa_pairs

# Path to your dataset
dataset_path = "../data/BioASQ-train-factoid-6b-full-annotated.json"
documents, _ = load_bioasq_dataset(dataset_path)

# Initialize embeddings using HuggingFaceEmbeddings (base model)
print("Initializing embeddings using HuggingFaceEmbeddings (all-mpnet-base-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Create a vector store for retrieval (FAISS for dense retrieval)
print("Creating vector store from documents using FAISS...")
vectorstore = FAISS.from_documents(documents, embeddings)

# Define a directory to save the vector store
save_dir = "../vector_store"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save the vector store to disk
print(f"Saving vector store to directory: {save_dir}")
vectorstore.save_local(save_dir)
print("Vector store saved successfully.")
