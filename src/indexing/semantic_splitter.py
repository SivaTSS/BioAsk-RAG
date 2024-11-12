# src/indexing/semantic_splitter.py

from langchain.text_splitter import RecursiveCharacterTextSplitter

class SemanticSplitter:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def split(self, text):
        return self.splitter.split_text(text)
