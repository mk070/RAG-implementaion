# chunking.py
from langchain_text_splitters  import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def create_chunks(self, text):
        chunks = self.text_splitter.split_text(text)
        return chunks
