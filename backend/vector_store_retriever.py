# vector_store_retriever.py

class VectorStoreRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve_documents(self, query, top_k=1):
        return self.vectorstore.similarity_search(query, k=top_k)
