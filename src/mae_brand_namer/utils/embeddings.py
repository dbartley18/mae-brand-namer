"""Embedding utilities for the brand naming workflow."""

import os
from typing import List
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Configure Google API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using Google's text embeddings.
    
    Args:
        texts (List[str]): List of texts to embed
        
    Returns:
        List[List[float]]: List of embeddings, each embedding is a list of floats
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        task_type="retrieval_document"  # Optimized for document/passage retrieval
    )
    
    # Get embeddings for all texts
    embedded = embeddings.embed_documents(texts)
    
    return embedded 