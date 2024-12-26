# embedding_generator.py

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS

class EmbeddingGenerator:
    def __init__(self, model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def generate_embeddings(self, documents):
        return FAISS.from_documents(documents, self.embedding_model)
