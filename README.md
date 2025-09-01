Medical AI Assistant — Technical Documentation
Table of Contents
Introduction

Setup

Architecture Overview

Index Configuration & Vector Database

Embedding and Chunking

Retriever and Reranker

Large Language Model and Answering

Frontend

Appendix

Introduction
The Medical AI Assistant is an AI-powered system designed to answer domain-specific medical queries using contextual information extracted from a knowledge base of medical documents. It combines document embeddings, vector search, and a powerful language model to provide accurate, contextually grounded responses, delivered through an interactive web-based interface.

Setup
Environment variables: Load from .env file using dotenv.

API Tokens: HuggingFace API token stored in the environment variable HF_TOKEN.

Dependencies: Python packages including langchain, langchain_community, transformers, streamlit, and others required for embeddings, vector stores, and LLM interaction.

Vector database location: Stored locally at vectorstore/db_faiss.

Data Directory: Medical PDF documents stored in Data/ folder for ingestion.

Architecture Overview
The system architecture consists of the following key components:

Document Loader and Chunker: Loads medical PDFs and splits the content into chunks.

Embedding Model: Generates dense vector embeddings for each text chunk.

Vector Database (FAISS): Stores the vector embeddings for similarity search.

Retriever (FAISS): Searches the vector database for relevant chunks matching a user query.

Large Language Model (LLM): Meta-LLaMA 3 8B model accessed via HuggingFace endpoint, responsible for generating answers.

RetrievalQA Chain: Orchestrates retrieval of context and LLM response generation with a custom prompt.

Frontend: Streamlit web app that manages user interaction, query submission, and displays results.

Index Configuration & Vector Database
The documents are loaded from a local Data/ directory containing PDF files.

PDF documents are ingested using PyPDFLoader and DirectoryLoader.

Text is split into chunks of 400 characters with 50 characters overlap to maintain context across chunks.

Each chunk is converted into embeddings using the pre-trained model sentence-transformers/all-MiniLM-L6-v2.

The embeddings are stored in a FAISS vector store saved locally at vectorstore/db_faiss.

The vector database supports fast similarity search during query time, retrieving relevant document chunks.

Embedding and Chunking
Document Loading: Utilizes Langchain's PDF loaders for batch loading.

Text Splitting: RecursiveCharacterTextSplitter splits documents for manageable semantic chunks.

Embedding Model: Uses HuggingFace Embeddings based on sentence-transformers/all-MiniLM-L6-v2 configured with normalized embeddings on CPU.

This approach balances chunk size and overlap to preserve semantic coherence in retrieved results.

Retriever and Reranker
The FAISS vector store acts as the retriever by performing approximate nearest neighbor search to find the top 5 most relevant document chunks for a given query.

A RetrievalQA chain integrates the retriever with the LLM.

A custom prompt template ensures the model:

Only answers based on retrieved context.

Provides clear and detailed responses.

Includes citations linked to source documents.

The retriever enforces strict grounding of answers to prevent hallucination or out-of-context responses.

Large Language Model and Answering
The LLM is sourced from the HuggingFace model repo meta-llama/Meta-Llama-3-8B and accessed via API endpoint.

Key LLM configuration:

Temperature: 0.1 to 0.5 for controlled output variability.

Max tokens: 512 to limit response length.

The QA chain uses a prompt template requiring the model to answer medical questions strictly from provided context.

The output consists of:

A well-structured, clear answer paragraph.

Explicit citations referencing source documents used for answering.

Frontend
Developed in Streamlit for rapid UI development and deployment.

Maintains session state for chat history and current responses.

UI flow:

User inputs a medical query.

The backend retrieves documents and generates an answer.

The retrieved documents and their citations are displayed alongside the answer.

It validates retrieved documents for medical relevance before generating responses, enhancing answer accuracy and appropriateness.

Appendix
Environment setup: Install Python dependencies via requirements.txt or pip.

Running the project: Execute the Streamlit app frontend script to launch the web interface.

Data ingestion: Use the provided script for loading and indexing PDF documents.

Token management: Keep HuggingFace API tokens secure and update .env as needed.

This documentation provides a clear, structured overview of the Medical AI Assistant project's setup, components, and workflow to facilitate understanding, usage, and future enhancements.
