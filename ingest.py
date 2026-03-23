# ingest.py - Document Ingestion Script

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Try new import path first, fall back to old
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file! Please add your API key.")

print("✅ API Key loaded successfully")

# File path
pdf_path = "WabagTech-FY26Q3.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"❌ PDF not found at {pdf_path}")

print(f"✅ Found PDF: {pdf_path}")

# Load PDF
print("📄 Loading PDF...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} pages")

# Chunk text
print("✂️  Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# Create embeddings
print("🧠 Initializing embedding model...")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)
print("✅ Embedding model ready")

# Create vector database
print("💾 Creating vector database...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="wabag_earnings"
)
print("✅ Vector database created!")
print(f"📍 Database location: ./chroma_db/")

# Test retrieval
print("\n🔍 Testing retrieval...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
test_query = "What was the revenue growth?"
results = retriever.invoke(test_query)
print(f"✅ Retrieval test successful! Found {len(results)} relevant chunks")

print("\n" + "="*70)
print("🎉 INGESTION COMPLETE!")
print("="*70)
print(f"✅ Processed: {len(documents)} pages")
print(f"✅ Created: {len(chunks)} chunks")
print(f"✅ Ready for querying!")
print("="*70)