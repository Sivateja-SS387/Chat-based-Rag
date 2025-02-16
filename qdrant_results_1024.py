import logging
import traceback
from typing import List, Dict, Any
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import time
import streamlit as st
import os

# Set Streamlit page config first
st.set_page_config(page_title="Pharma Chat Assistant", layout="centered")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "Drug_Data_Collection"
TOP_K = 5  # Number of results to retrieve
SAMBANOVA_API_URL = "https://api.sambanova.ai/v1/chat/completions"
SAMBANOVA_API_KEY = "3a32e1a5-667a-4929-a146-fdabc9b7abb9"

# Initialize resources
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
generator = pipeline("text2text-generation", model="t5-small")

# Initialize Qdrant client with health check
try:
    # qdrant_host = os.getenv('QDRANT_HOST', 'qdrant')  # Use container name as default host
    # qdrant_port = int(os.getenv('QDRANT_PORT', '6334'))  # Use internal port
    # qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

    

    qdrant_client = QdrantClient(
        url="https://1ca397e9-f73f-403d-a49e-17db01f4ce1a.us-east4-0.gcp.cloud.qdrant.io:6333", 
        api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ3NDYyNzQ0fQ.aZe8GuXmOTNyNDjmC3nz3bFr3ozJu3zTMiA0qySiQKo",
    )


    collections = qdrant_client.get_collections()
    if not any(col.name == COLLECTION_NAME for col in collections.collections):
        logger.warning(f"Collection {COLLECTION_NAME} not found in Qdrant")
    logger.info("Successfully connected to Qdrant and verified collection")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {e}")
    st.error(f"Failed to connect to Qdrant: {e}")

def get_query_embedding(query: str) -> np.ndarray:
    try:
        embedding = hf_embeddings.embed_query(query)
        return embedding
    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        raise

def retrieve_from_qdrant(
    query_embedding: np.ndarray,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    try:
        start_time = time.time()
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        end_time = time.time()
        logger.info(f"Qdrant retrieval time: {end_time - start_time:.4f} seconds")
        return [
            {
                "score": hit.score,
                "payload": hit.payload
            }
            for hit in results
        ]
    except Exception as e:
        logger.error(f"Error during Qdrant retrieval: {e}")
        traceback.print_exc()
        return []

def query_sambanova(prompt: str) -> str:
    try:
        headers = {"Authorization": f"Bearer {SAMBANOVA_API_KEY}"}
        payload = {
            "model": "Meta-Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that provides structured and detailed responses about drug information."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 500
        }
        response = requests.post(SAMBANOVA_API_URL, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json().get("choices")[0].get("message").get("content")
        else:
            raise Exception(f"SambaNova API Error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error during SambaNova query: {e}")
        traceback.print_exc()
        return ""

def rag_pipeline(query: str) -> Dict[str, Any]:
    try:
        # Step 1: Get query embedding
        query_embedding = get_query_embedding(query)

        # Step 2: Retrieve results from Qdrant
        retrieved_docs = retrieve_from_qdrant(query_embedding)
        if not retrieved_docs:
            return {
                "query": query,
                "response": "No relevant information found.",
                "retrieved_docs": []
            }

        # Step 3: Prepare context for Sambanova
        # context = " ".join([doc["payload"].get("description", "") for doc in retrieved_docs])
        # Step 3: Prepare context for Sambanova
        context = " ".join([str(doc["payload"]) for doc in retrieved_docs])

        print(context)
        # Step 4: Generate a structured prompt for Sambanova
        # final_prompt = f"Based on the following context only, provide a detailed response to the query.\n\nContext: {context}\nQuery: {query}\n\nResponse: if for the query the contect dont have match dnt dorrelat and dnt get from any sources" 
        final_prompt = f"Based on the following context only, provide a detailed framed response to the query.\n\nContext: {context}\nQuery: {query}\n\nResponse:  Do not use any external sources or assumptions beyond the provided context.and in output dnt mention these are details i can answer. " 

        # Step 5: Get response from Sambanova
        structured_response = query_sambanova(final_prompt)

        return {
            "query": query,
            "response": structured_response,
            "retrieved_docs": retrieved_docs
        }
    except Exception as e:
        logger.error(f"Error in rag_pipeline: {e}")
        traceback.print_exc()
        return {"query": query, "response": "Error processing the query."}

# Streamlit UI
st.title("💊 Pharmacy Chat Assistant")
# st.write("Ask questions about drug data. The assistant responds based on the database.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["sender"]):
        st.markdown(message["text"])

# Input for user messages
if user_input := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"sender": "user", "text": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve context dynamically from user input
    retrieved_context = rag_pipeline(user_input)
    # Prepare the prompt for SambaNova
    # final_prompt = f"Answer the following based on context:\nContext: {retrieved_context['response']}\nQuestion: {user_input}"

    # print(final_prompt)
    # Get response from SambaNova
    # response = query_sambanova(final_prompt)
    print(retrieved_context['response'])

    # Add bot response to chat history
    st.session_state.messages.append({"sender": "bot", "text": retrieved_context['response']})

    # Display bot response
    with st.chat_message("bot"):
        st.markdown(retrieved_context['response'])
