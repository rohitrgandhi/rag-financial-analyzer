# query.py - Simplified RAG Query System

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# ============================================================================
# SETUP
# ============================================================================

print("🚀 Starting RAG Query System...")
print("="*70)

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found!")

print("✅ API Key loaded")

# ============================================================================
# LOAD VECTOR DATABASE
# ============================================================================

print("📂 Loading vector database...")

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)

# Load the vector database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="wabag_earnings"
)

print("✅ Vector database loaded!")

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=api_key
)

print("✅ Ready to answer questions!")
print("="*70)
print("\n💬 Ask me anything about the Wabag earnings call!")
print("Type 'quit' to exit\n")

# ============================================================================
# QUERY LOOP
# ============================================================================

while True:
    question = input("❓ Your question: ").strip()
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("\n👋 Goodbye!")
        break
    
    if not question:
        continue
    
    print("\n🔍 Searching...")
    
    try:
        # Retrieve relevant chunks
        docs = retriever.invoke(question)
        
        # Combine chunks into context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""You are a financial analyst assistant analyzing VA Tech Wabag's Q3 FY26 earnings call.

Use the following context to answer the question. If you cannot find the answer, say "I cannot find this information in the transcript."

Context:
{context}

Question: {question}

Answer:"""
        
        # Get answer from LLM
        print("🤖 Generating answer...\n")
        response = llm.invoke(prompt)
        
        # Print answer
        print("="*70)
        print("📝 ANSWER:")
        print("="*70)
        print(response.content)
        print("="*70)
        
        # Show sources
        print("\n📚 Sources used:")
        for i, doc in enumerate(docs, 1):
            print(f"\nSource {i}: {doc.page_content[:150]}...")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")

print("\n✅ Session ended.")