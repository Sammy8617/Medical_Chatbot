<div align="center">
  <h1>Medical AI Assistant</h1>
  <p><b>Project Documentation</b></p>
</div>

---

# 📚 Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Setup & Installation](#setup--installation)
4. [Architecture Overview](#architecture-overview)
5. [Workflow](#workflow)
6. [Frontend](#frontend)
7. [Appendix](#appendix)

---

## 🩺 Introduction

Medical AI Assistant is an interactive, AI-powered chatbot designed to answer medical queries using information extracted from a curated set of medical documents. It leverages state-of-the-art language models, document embeddings, and vector search to deliver accurate, context-aware responses through a user-friendly web interface.

---

## ✨ Features

- Ingests and indexes medical PDFs for domain-specific knowledge.
- Uses advanced embedding models for semantic search.
- Retrieves contextually relevant document chunks for each query.
- Integrates with large language models (LLMs) for answer generation.
- Provides citations and source references for transparency.
- Streamlit-based frontend for easy interaction.

---

## ⚙️ Setup & Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/Sammy8617/Medical_Chatbot.git
    cd Medical_Chatbot/Chatbot_architecture
    ```
2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3. **Configure environment variables**
    - Add your HuggingFace API token to `.env`:
      ```env
      HF_TOKEN=your_huggingface_token
      ```
4. **Prepare data**
    - Place medical PDF files in the `Data/` directory.
5. **Run the application**
    ```bash
    streamlit run frontend.py
    ```

---

## 🏗️ Architecture Overview

**Main Components:**

- **Document Loader & Chunker:** Loads PDFs and splits text into semantic chunks.
- **Embedding Model:** Generates vector embeddings (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
- **Vector Database (FAISS):** Stores embeddings for fast similarity search.
- **Retriever:** Finds top relevant chunks for each query.
- **LLM (e.g., Meta-LLaMA, GPT-2):** Generates answers based on retrieved context.
- **RetrievalQA Chain:** Orchestrates retrieval and answer generation.
- **Frontend:** Streamlit app for user interaction.

---

## 🔄 Workflow

1. **Document Ingestion:**
    - PDFs loaded from `Data/` using LangChain loaders.
    - Text split into ~400 character chunks with overlap.
2. **Embedding & Indexing:**
    - Chunks embedded using HuggingFace models.
    - Embeddings stored in FAISS vector store (`vectorstore/db_faiss`).
3. **Query Processing:**
    - User submits a query via the frontend.
    - Retriever finds top 5 relevant chunks.
    - LLM answers using only retrieved context (custom prompt enforces this).
    - Citations and source references included in the response.

---

## 🖥️ Frontend

- Built with Streamlit for rapid prototyping and deployment.
- Features:
  - Chat interface for medical queries.
  - Displays answers and source citations.
  - Maintains session history.

---

## 📎 Appendix

- **Environment setup:** Install dependencies via `requirements.txt`.
- **Data ingestion:** Use provided scripts to index new PDFs.
- **Token management:** Keep your HuggingFace API token secure in `.env`.
- **Troubleshooting:**
  - Ensure your model supports `text-generation` for LLM tasks.
  - Check permissions for private models.
  - Use public models like `gpt2` for testing.

---

<div align="center">⁂</div>

## Architecture Overview

The system architecture consists of the following key components:

- **Document Loader and Chunker**: Loads medical PDFs and splits the content into chunks.
- **Embedding Model**: Generates dense vector embeddings for each text chunk.
- **Vector Database (FAISS)**: Stores the vector embeddings for similarity search.
- **Retriever (FAISS)**: Searches the vector database for relevant chunks matching a user query.
- **Large Language Model (LLM)**: Meta-LLaMA 3 8B model accessed via HuggingFace endpoint, responsible for generating answers.
- **RetrievalQA Chain**: Orchestrates retrieval of context and LLM response generation with a custom prompt.
- **Frontend**: Streamlit web app that manages user interaction, query submission, and displays results.

***

## Index Configuration \& Vector Database

- The documents are loaded from a local `Data/` directory containing PDF files.
- PDF documents are ingested using `PyPDFLoader` and `DirectoryLoader`.
- Text is split into chunks of **400 characters** with **50 characters overlap** to maintain context across chunks.
- Each chunk is converted into embeddings using the pre-trained model `sentence-transformers/all-MiniLM-L6-v2`.
- The embeddings are stored in a FAISS vector store saved locally at `vectorstore/db_faiss`.
- The vector database supports fast similarity search during query time, retrieving relevant document chunks.

***

## Embedding and Chunking

- **Document Loading**: Utilizes Langchain's PDF loaders for batch loading.
- **Text Splitting**: `RecursiveCharacterTextSplitter` splits documents for manageable semantic chunks.
- **Embedding Model**: Uses HuggingFace Embeddings based on `sentence-transformers/all-MiniLM-L6-v2` configured with normalized embeddings on CPU.
- This approach balances chunk size and overlap to preserve semantic coherence in retrieved results.

***

## Retriever and Reranker

- The FAISS vector store acts as the retriever by performing approximate nearest neighbor search to find the **top 5 most relevant document chunks** for a given query.
- A `RetrievalQA` chain integrates the retriever with the LLM.
- A custom prompt template ensures the model:
    - Only answers based on retrieved context.
    - Provides clear and detailed responses.
    - Includes citations linked to source documents.
- The retriever enforces strict grounding of answers to prevent hallucination or out-of-context responses.

***

## Large Language Model and Answering

- The LLM is sourced from the HuggingFace model repo `meta-llama/Meta-Llama-3-8B` and accessed via API endpoint.
- Key LLM configuration:
    - Temperature: 0.1 to 0.5 for controlled output variability.
    - Max tokens: 512 to limit response length.
- The QA chain uses a prompt template requiring the model to answer medical questions strictly from provided context.
- The output consists of:
    - A well-structured, clear answer paragraph.
    - Explicit citations referencing source documents used for answering.

***

## Frontend

- Developed in Streamlit for rapid UI development and deployment.
- Maintains session state for chat history and current responses.
- UI flow:
    - User inputs a medical query.
    - The backend retrieves documents and generates an answer.
    - The retrieved documents and their citations are displayed alongside the answer.
- It validates retrieved documents for medical relevance before generating responses, enhancing answer accuracy and appropriateness.

***

## Appendix

- **Environment setup**: Install Python dependencies via `requirements.txt` or pip.
- **Running the project**: Execute the Streamlit app frontend script to launch the web interface.
- **Data ingestion**: Use the provided script for loading and indexing PDF documents.
- **Token management**: Keep HuggingFace API tokens secure and update `.env` as needed.

***
<img width="1920" height="1080" alt="Screenshot 2025-09-01 215357" src="https://github.com/user-attachments/assets/32eeb14d-c2c5-4241-b7bb-4d3696224dbc" />
<img width="1920" height="1080" alt="Screenshot 2025-09-01 215317" src="https://github.com/user-attachments/assets/b3f8ebdc-95ff-4751-9a2f-925f34087aa7" />
<img width="1920" height="1080" alt="Screenshot 2025-09-01 222228" src="https://github.com/user-attachments/assets/6de1948c-2928-4a6e-8996-d0e00a4384a8" />
<img width="1920" height="1080" alt="Screenshot 2025-09-01 231705" src="https://github.com/user-attachments/assets/90a20992-8183-4c55-b688-423e2b2f34d2" />

