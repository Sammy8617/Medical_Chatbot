from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

HF_TOKEN=os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID="meta-llama/Meta-Llama-3-8B"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512
    )
    return llm

CUSTOM_PROMPT_TEMPLATE="""You are a helpful medical assistant. Use the following context to answer the user's question.

Context: {context}

Question: {question}

Instructions:
1. First, provide a clear and detailed answer based on the context
2. Write your response in a well-structured paragraph
3. At the end, include citations [1], [2], [3], etc. for the sources you used
4. If you don't know the answer, simply say "I don't know the answer to this question based on the provided context."

Answer:"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Check if the vector database exists
import os
if os.path.exists(DB_FAISS_PATH):
    print(f"Loading existing vector database from: {DB_FAISS_PATH}")
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print(f"Vector database loaded successfully. Number of documents: {len(db.index_to_docstore_id)}")
else:
    print(f"Vector database not found at: {DB_FAISS_PATH}")
    print("You need to create the vector database first by ingesting documents.")
    exit(1)

qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k":5}),
    chain_type_kwargs={"prompt":set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
    return_source_documents=True
)

user_query=input("Write Query here")

# First, let's see what documents are retrieved
print("\n" + "="*50)
print("RETRIEVING DOCUMENTS...")
print("="*50)
retrieved_docs = db.as_retriever(search_kwargs={"k":5}).get_relevant_documents(user_query)
print(f"Retrieved {len(retrieved_docs)} documents")

for i, doc in enumerate(retrieved_docs, 1):
    print(f"\nDocument {i}:")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content preview: {doc.page_content[:150]}...")

print("\n" + "="*50)
print("GENERATING ANSWER...")
print("="*50)

responses=qa_chain.invoke({'query':user_query})
print("RESULT:",responses['result'])

# Check if source_documents exist and display them with citations
if 'source_documents' in responses and responses['source_documents']:
    print("\n" + "="*50)
    print("SOURCE DOCUMENTS (Citations):")
    print("="*50)
    
    for i, doc in enumerate(responses['source_documents'], 1):
        print(f"\n[{i}] Source: {doc.metadata.get('source', 'Unknown source')}")
        print(f"Page: {doc.metadata.get('page', 'Unknown page')}")
        print(f"Content: {doc.page_content[:300]}...")  # Show first 300 characters
        if len(doc.page_content) > 300:
            print("...")
else:
    print("\nNo source documents found in response")
    print("Response keys available:", list(responses.keys()))
    print("Full response structure:", responses)  