import os
import sys
import json
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline

def initialize_llm():
    # Set device to -1 for CPU usage
    device = 0 if torch.cuda.is_available() else -1
    print("Initializing local language model using HuggingFacePipeline (google/flan-t5-base) on CPU...")
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=device,
        max_new_tokens=100,  # Adjust as needed for longer outputs
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print("Local language model initialized.")
    return llm

def load_all_questions(file_path="../data/BioASQ-train-factoid-6b-full-annotated.json"):
    print("Loading questions from dataset:", file_path)
    questions = []
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return questions

    # Collect all questions from the dataset
    for entry in data.get('data', []):
        for paragraph in entry.get('paragraphs', []):
            for qa in paragraph.get('qas', []):
                question = qa.get('question')
                if question:
                    questions.append((question, qa.get('answers')))
    print(f"Loaded {len(questions)} questions from dataset.")
    return questions

def main():
    # Initialize the local language model with an instruction-tuned, CPU-friendly model.
    llm = initialize_llm()

    # Define a simple prompt template for direct Q&A (without any context)
    prompt_template = """You are a helpful assistant.
Answer the following question directly and concisely.
Question: {question}
Answer:"""
    simple_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question"]
    )
    
    # Determine the questions to answer:
    questions = []
    if len(sys.argv) > 1:
        # If questions are provided as command-line arguments, treat them as a single question
        questions.append((" ".join(sys.argv[1:]), None))
        print("Using provided question:", questions[0][0])
    else:
        # Load all questions from the dataset
        all_questions = load_all_questions()
        if all_questions:
            # Select every 100th question and take up to 3 questions
            selected = all_questions[::100][:3]
            questions.extend(selected)
            print("Using selected questions from dataset:")
            for idx, (q, gt) in enumerate(questions, start=1):
                print(f"{idx}. Question: {q}")
                print(f"   Ground Truth: {gt}")
        else:
            # Fallback to interactive input if dataset loading fails
            user_q = input("Enter your question: ")
            questions.append((user_q, None))

    print("\n--- Running Direct QA ---")
    # Process each question
    for question, ground_truth in questions:
        prompt = simple_prompt.format(question=question)
        answer = llm.invoke(prompt)
        print("\n===================================")
        print("Question:", question)
        print("Generated Answer:", answer)
        if ground_truth:
            print("Ground Truth:", ground_truth)
        print("===================================\n")

if __name__ == "__main__":
    main()
