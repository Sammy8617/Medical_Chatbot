from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import os

DATA_PATH="Data/"
def load_pdf_files(data):
    loader= DirectoryLoader(
        data,
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)

def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=400,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)

def get_embeddings_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence_transformers/all-Mini-L6-v2")
    return embedding_model

embedding_model=get_embeddings_model()

DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)