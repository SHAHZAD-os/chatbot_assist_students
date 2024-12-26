from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.main import RAGSystem  # Ensure the correct import of RAGSystem from main.py
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Add CORS middleware to allow requests from the Flutter frontend (or any domain)
origins = [
    "http://localhost:53316",  # Flutter frontend
    "http://localhost:8000",    # FastAPI server URL (localhost:8000)
    "http://localhost:53520",   # Add more origins if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow your Flutter app's URL to access the FastAPI app
    allow_credentials=True,
    allow_methods=["*"],    # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)

# Define the request model
class QueryRequest(BaseModel):
    query: str

# Initialize the RAG system with the path to your document
filepath = r"C:\Users\HB LAPTOP POINT\Desktop\chatbot_assist_students\backend\BS Curriculm Computing Disciplines-2023.pdf"
rag_system = RAGSystem(filepath)


def remove_duplicates(context):
    sentences = context.split(". ")
    unique_sentences = []

    seen = set()

    for sentence in sentences:
        if sentence not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence)

    cleaned_responce=('. ').join(unique_sentences)
    return cleaned_responce

@app.post("/query/")
async def query_rag(query_request: QueryRequest):
    """
    Endpoint to receive a query and return the answer from the RAG system.
    """
    try:
        query = query_request.query
        
        # Log the query for debugging purposes
        logging.info(f"Received query: {query}")

        # Use asyncio to avoid blocking the event loop if `answer_query` is blocking
        answer = await asyncio.to_thread(rag_system.answer_query, query)

        cleaned_answer = remove_duplicates(answer)

        # Log the answer before sending it
        logging.info(f"Generated answer: {cleaned_answer}")
        
        return {"answer": cleaned_answer}
    
    except Exception as e:
        logging.error(f"Error occurred while processing the query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the query: {str(e)}")

