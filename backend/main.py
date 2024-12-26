# main.py

import os
from .loader_file import DocumentLoader
from .text_splitter import TextSplitter
from .embedding import EmbeddingGenerator
from .vector_store_retriever import VectorStoreRetriever
from .llm_handler import LLMHandler

class RAGSystem:
    def __init__(self, filepath):
        self.document_loader = DocumentLoader(filepath)
        self.text_splitter = TextSplitter()
        self.embedding_generator = EmbeddingGenerator()
        self.llm_handler = LLMHandler()
        self.vectorstore = None
        self.setup()

    def setup(self):
        # Load and preprocess documents
        documents = self.document_loader.load_documents()
        split_documents = self.text_splitter.split_text(documents)
        
        # Generate embeddings and create vector store
        self.vectorstore = self.embedding_generator.generate_embeddings(split_documents)

    def answer_query(self, query):
        # Retrieve relevant documents based on the query
        retrieved_docs = self.vectorstore.similarity_search(query)
        document_content = retrieved_docs[0].page_content if retrieved_docs else "No relevant documents found."
        
        # Generate the answer using the LLM
        return self.llm_handler.generate_response(query, document_content)
    



