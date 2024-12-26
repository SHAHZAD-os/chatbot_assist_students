# document_loader.py

from langchain_community.document_loaders import PyPDFLoader

class DocumentLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_documents(self):
        loader = PyPDFLoader(self.filepath)
        documents = loader.load()
        if not documents:
            raise Exception("No documents loaded.")
        return documents
