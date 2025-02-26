import os
import sys
import json
import random
import torch
import gradio as gr
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# ----------------------------------------------------------------
# Utility: Load sample questions with ground truth from dataset
# ----------------------------------------------------------------
def load_sample_questions(file_path="../data/BioASQ-train-factoid-6b-full-annotated.json"):
    sample_dict = {}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return sample_dict
    for entry in data.get('data', []):
        for paragraph in entry.get('paragraphs', []):
            for qa in paragraph.get('qas', []):
                question = qa.get('question')
                answers = qa.get('answers')
                if question and answers:
                    gt = ", ".join([ans["text"] if isinstance(ans, dict) and "text" in ans else str(ans) for ans in answers])
                    sample_dict[question] = gt
    return sample_dict

# Global sample questions dictionary
SAMPLE_Q_DICT = load_sample_questions()
if not SAMPLE_Q_DICT:
    SAMPLE_Q_DICT = {
        "What is the capital of France?": "Paris",
        "Who wrote '1984'?": "George Orwell",
        "What is the boiling point of water?": "100°C",
        "What is the chemical formula of water?": "H2O",
        "Who painted the Mona Lisa?": "Leonardo da Vinci",
        "What is the largest planet in our solar system?": "Jupiter",
        "Who discovered penicillin?": "Alexander Fleming",
        "What is the speed of light?": "Approximately 299,792 km/s",
        "What is the currency of Japan?": "Yen",
        "Who is known as the father of computers?": "Charles Babbage"
    }

# Use at least 10 sample questions
SAMPLE_Q_LIST = list(SAMPLE_Q_DICT.keys())[:10]

# ----------------------------------------------------------------
# Utility: Initialize a language model (CPU-friendly)
# ----------------------------------------------------------------
def initialize_llm(model_name):
    device = 0 if torch.cuda.is_available() else -1
    print(f"Initializing model {model_name} on device {device}...")
    pipe = pipeline(
        "text2text-generation",
        model=model_name,
        device=device,
        max_new_tokens=100,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# ----------------------------------------------------------------
# Direct Q&A (no retrieval)
# ----------------------------------------------------------------
def direct_qa(query, model_name):
    llm = initialize_llm(model_name)
    prompt_template = (
        "You are a helpful assistant.\n"
        "Answer the following question directly and concisely.\n"
        "Question: {question}\n"
        "Answer:"
    )
    simple_prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    prompt = simple_prompt.format(question=query)
    answer = llm.invoke(prompt)
    return answer

# ----------------------------------------------------------------
# Retrieval-Augmented Generation (RAG) Q&A
# ----------------------------------------------------------------
def rag_qa(query, model_name):
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vector_store_dir = "../vector_store"
    vectorstore = FAISS.load_local(vector_store_dir, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = initialize_llm(model_name)
    prompt_template = (
        "You are a helpful assistant.\n"
        "Use the following context to answer the question.\n"
        "If the answer is not contained in the context, say \"I don't know.\"\n"
        "Context:\n{context}\n"
        "Question: {question}\n"
        "Answer:"
    )
    qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt}
    )
    answer = qa_chain.invoke(query)
    return answer

# ----------------------------------------------------------------
# Main answer function: Choose mode, show ground truth, and format output.
# ----------------------------------------------------------------
def answer_question(query, qa_mode, model_name, sample_choice):
    if not query.strip():
        if sample_choice and sample_choice != "None":
            query = sample_choice
        else:
            query = random.choice(SAMPLE_Q_LIST)
    print(f"Final query: {query}")
    ground_truth = SAMPLE_Q_DICT.get(query, "Not available")
    if qa_mode == "RAG Q&A":
        answer = rag_qa(query, model_name)
    else:
        answer = direct_qa(query, model_name)
    # If answer is a dict (as in RAG mode), extract only the 'result' field.
    if isinstance(answer, dict) and "result" in answer:
        answer = answer["result"]
    result = f"""
    <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
      <h3>Answer</h3>
      <p style="font-size: 16px;">{answer}</p>
    </div>
    <div style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px;">
      <h4>Ground Truth</h4>
      <p style="font-size: 14px;">{ground_truth}</p>
    </div>
    """
    return result

# ----------------------------------------------------------------
# Gradio Interface (using Gradio default styling)
# ----------------------------------------------------------------
def main_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## BioAsk Q&A System")
        gr.Markdown("Select your QA mode, choose a model, and either type your question or pick a sample question. The ground truth is provided for verification.")
        
        with gr.Row():
            qa_mode = gr.Radio(
                choices=["Direct Q&A", "RAG Q&A"],
                label="Select QA Mode",
                value="Direct Q&A",
                info="Direct Q&A answers without external context; RAG Q&A retrieves context from a vector store."
            )
            model_dropdown = gr.Dropdown(
                choices=["google/flan-t5-base", "google/flan-t5-small", "t5-small", "t5-base", "facebook/bart-base"],
                label="Select Model",
                value="google/flan-t5-base",
                info="Select a CPU-friendly model to compare performance."
            )
        with gr.Row():
            query_input = gr.Textbox(lines=2, placeholder="Enter your question here...", label="Your Question")
            sample_dropdown = gr.Dropdown(
                choices=["None"] + SAMPLE_Q_LIST,
                label="Or select a sample question",
                value="None",
                info="Select one of at least 10 curated sample questions."
            )
        with gr.Row():
            output = gr.HTML(label="Generated Answer")
        with gr.Row():
            submit_btn = gr.Button("Submit")
            reset_btn = gr.Button("Reset")
        
        submit_btn.click(
            fn=answer_question,
            inputs=[query_input, qa_mode, model_dropdown, sample_dropdown],
            outputs=output
        )
        
        def reset_all():
            return "", "None", ""
        reset_btn.click(
            fn=reset_all,
            inputs=None,
            outputs=[query_input, sample_dropdown, output]
        )
    demo.launch()

if __name__ == "__main__":
    main_interface()
