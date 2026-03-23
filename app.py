# app.py - Streamlit Web Interface for RAG Financial Analyzer

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# Page configuration
st.set_page_config(
    page_title="RAG Financial Analyzer",
    page_icon="📊",
    layout="wide"
)

# Load environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found in .env file!")
    st.stop()

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize RAG components (cached for performance)
@st.cache_resource
def load_rag_system():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="wabag_earnings"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=api_key
    )
    
    return retriever, llm

# Load RAG system
try:
    retriever, llm = load_rag_system()
except Exception as e:
    st.error(f"❌ Error loading RAG system: {e}")
    st.info("💡 Make sure you've run `python ingest.py` first!")
    st.stop()

# Header
st.title("📊 RAG Financial Document Analyzer")
st.markdown("**Ask questions about VA Tech Wabag Q3 FY26 Earnings Call**")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This RAG system analyzes the **VA Tech Wabag Q3 & 9M FY26 Earnings Call** transcript.
    
    **📈 System Metrics:**
    - **Documents:** 23 pages
    - **Chunks:** 91 segments
    - **Embeddings:** 1,536 dimensions
    - **Evaluation Score:** 0.79
    
    **🎯 Evaluation Metrics:**
    - Faithfulness
    - Answer Relevancy
    - Context Precision
    - Context Recall
    """)
    
    st.divider()
    
    st.header("💡 Sample Questions")
    
    sample_questions = [
        "What was the revenue in Q3 FY26?",
        "What was the EBITDA margin?",
        "What is the net cash position?",
        "Who is the CEO of India Cluster?",
        "What was the PAT growth year-on-year?"
    ]
    
    for q in sample_questions:
        if st.button(q, key=f"sample_{q}"):
            st.session_state.sample_question = q
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.markdown("""
    **🔗 Links:**
    - [GitHub Repository](https://github.com/rohitrgandhi/rag-financial-analyzer)
    - [Project Documentation](https://github.com/rohitrgandhi/rag-financial-analyzer#readme)
    """)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📄 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(source[:300] + "..." if len(source) > 300 else source)
                    st.divider()

# Handle sample question from sidebar
if 'sample_question' in st.session_state:
    question = st.session_state.sample_question
    del st.session_state.sample_question
else:
    question = st.chat_input("Ask a question about the earnings call...")

# Process user input
if question:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.markdown(question)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents..."):
            # Retrieve context
            docs = retriever.invoke(question)
            contexts = [doc.page_content for doc in docs]
            
            # Generate answer
            context_text = "\n\n".join(contexts)
            prompt = f"""You are a financial analyst. Use this context to answer the question.

Context: {context_text}

Question: {question}

Answer:"""
            
            response = llm.invoke(prompt)
            answer = response.content
            
            # Display answer
            st.markdown(answer)
            
            # Display sources
            if contexts:
                with st.expander("📄 View Sources"):
                    for i, source in enumerate(contexts, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source[:300] + "..." if len(source) > 300 else source)
                        st.divider()
    
    # Add assistant response to chat
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": contexts
    })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built with LangChain, ChromaDB, OpenAI GPT-3.5, and Streamlit | 
    <a href='https://github.com/rohitrgandhi/rag-financial-analyzer' target='_blank'>View on GitHub</a>
</div>
""", unsafe_allow_html=True)