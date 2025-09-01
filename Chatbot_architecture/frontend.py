import streamlit as st
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .source-doc {
        background-color: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = None

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "meta-llama/Meta-Llama-3-8B"
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def load_models():
    """Load the embedding model and LLM"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        llm = HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.1,
            max_new_tokens=512
        )
        
        return embedding_model, llm
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

@st.cache_resource
def load_vector_database():
    """Load the FAISS vector database"""
    try:
        if not os.path.exists(DB_FAISS_PATH):
            st.error(f"Vector database {DB_FAISS_PATH} cannot be uploaded due to the large size")
            return None
        
        embedding_model, _ = load_models()
        if embedding_model is None:
            return None
            
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
        return None

def create_qa_chain():
    """Create the QA chain"""
    try:
        embedding_model, llm = load_models()
        db = load_vector_database()
        
        if embedding_model is None or llm is None or db is None:
            return None
        
        custom_prompt_template = """SYSTEM: You are a medical AI assistant that can ONLY answer questions based on the provided context. You have NO access to external knowledge, general information, or pre-trained knowledge about any topics.

CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. ONLY use information from the context below
2. If the context is empty or doesn't contain relevant information, respond with: "I cannot answer this question based on the provided medical context."
3. If the question is not medical-related, respond with: "This question is not related to medical topics covered in my documents."
4. DO NOT use ANY knowledge about sports, celebrities, history, or any other topics
5. DO NOT make assumptions or provide information not explicitly in the context
6. If you don't know the answer from the context, say "I cannot answer this question based on the provided medical context."

Context: {context}

Question: {question}

Remember: You can ONLY use the context above. Any other information is forbidden.

Answer:"""

        prompt = PromptTemplate(
            template=custom_prompt_template, 
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

def validate_context(query, retrieved_docs):
    """Validate if the retrieved context is relevant to the query"""
    if not retrieved_docs:
        return False, "No relevant documents found"
    
    # Check if any document content is actually relevant
    query_lower = query.lower()
    relevant_docs = []
    
    for doc in retrieved_docs:
        content_lower = doc.page_content.lower()
        # Check if the document contains any medical terms or relevant content
        if any(term in content_lower for term in ['medical', 'health', 'disease', 'treatment', 'symptom', 'diagnosis', 'patient', 'clinical']):
            relevant_docs.append(doc)
    
    if not relevant_docs:
        return False, "Retrieved documents are not medical-related"
    
    return True, relevant_docs

def get_response(query):
    """Get response from the QA chain"""
    try:
        qa_chain = create_qa_chain()
        if qa_chain is None:
            return None, None
        
        # Get relevant documents first
        db = load_vector_database()
        if db is None:
            return None, None
            
        retrieved_docs = db.as_retriever(search_kwargs={"k": 5}).get_relevant_documents(query)
        
        # Validate context relevance
        is_relevant, validated_docs = validate_context(query, retrieved_docs)
        
        if not is_relevant:
            return {'result': f"I cannot answer this question based on the provided medical context. {validated_docs}"}, []
        
        # Get response from QA chain
        response = qa_chain.invoke({'query': query})
        
        return response, validated_docs
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")
        return None, None

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical AI Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)
        
        # Model info
        st.info(f"**Model:** {HUGGINGFACE_REPO_ID}")
        
        # Database status
        db = load_vector_database()
        if db:
            st.success(f"✅ Vector database loaded\n**Documents:** {len(db.index_to_docstore_id)}")
        else:
            st.error("❌ Vector database not available")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("1. Ask medical questions in the chat box below")
        st.markdown("2. The AI will search through your medical documents")
        st.markdown("3. Answers include citations to source documents")
        st.markdown("4. View source documents in the sidebar after each response")

    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat input
        user_query = st.chat_input("Ask your medical question here...")
        
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Get response
            with st.spinner("🤔 Thinking..."):
                response, source_docs = get_response(user_query)
            
            if response and 'result' in response:
                # Add assistant response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": response['result'],
                    "source_docs": source_docs
                })
                st.rerun()
            else:
                st.error("Sorry, I couldn't generate a response. Please try again.")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 AI Assistant:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show source documents if available
                if "source_docs" in message and message["source_docs"]:
                    with st.expander("📚 View Source Documents"):
                        for i, doc in enumerate(message["source_docs"], 1):
                            st.markdown(f"""
                            <div class="source-doc">
                                <strong>Source {i}:</strong> {doc.metadata.get('source', 'Unknown source')}<br>
                                <strong>Page:</strong> {doc.metadata.get('page', 'Unknown page')}<br>
                                <strong>Content:</strong> {doc.page_content[:200]}...
                            </div>
                            """, unsafe_allow_html=True)
    
    with col2:
        # Recent questions and quick actions
        st.markdown("**💡 Quick Actions**")
        
        if st.button("❓ Sample Questions", use_container_width=True):
            sample_questions = [
                "What are the symptoms of diabetes?",
                "How is hypertension treated?",
                "What causes heart disease?",
                "Explain the treatment for asthma"
            ]
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": "Here are some sample questions you can ask:\n\n" + "\n".join([f"• {q}" for q in sample_questions])
            })
            st.rerun()
        
        # Show recent questions
        if st.session_state.chat_history:
            st.markdown("**📝 Recent Questions**")
            user_questions = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
            for i, question in enumerate(user_questions[-5:], 1):
                st.markdown(f"{i}. {question[:50]}...")
        
        # System status
        st.markdown("---")
        st.markdown("**🔧 System Status**")
        
        # Check if models are loaded
        embedding_model, llm = load_models()
        if embedding_model and llm:
            st.success("✅ Models loaded")
        else:
            st.error("❌ Models not loaded")
        
        # Check database
        if db:
            st.success("✅ Database connected")
        else:
            st.error("❌ Database not connected")

if __name__ == "__main__":
    main()
