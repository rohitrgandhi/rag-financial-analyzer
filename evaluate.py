# evaluate.py - RAGAS Evaluation Framework

import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

print("="*70)
print("🎯 RAG SYSTEM EVALUATION - RAGAS FRAMEWORK")
print("="*70)

# Load environment
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found!")

print("\n✅ API Key loaded")

# Initialize RAG components
print("📂 Loading RAG system...")

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

print("✅ RAG system loaded")

# Load golden dataset
print("\n📊 Loading golden dataset...")

try:
    df = pd.read_csv("golden_dataset.csv")
    print(f"✅ Loaded {len(df)} test questions")
except FileNotFoundError:
    print("❌ Error: golden_dataset.csv not found!")
    print("Please create golden_dataset.csv in the project directory.")
    exit(1)

# Generate answers for each question
print("\n🤖 Generating answers for test questions...")
print("⏳ This will take 2-3 minutes...\n")

results = []

for idx, row in df.iterrows():
    question = row['question']
    ground_truth = row['ground_truth']
    
    print(f"Question {idx+1}/{len(df)}: {question[:50]}...")
    
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
    
    results.append({
        'question': question,
        'answer': answer,
        'contexts': contexts,
        'ground_truth': ground_truth
    })
    
    print(f"   ✓ Generated")

print("\n✅ All answers generated!")

# Convert to RAGAS dataset format
print("\n📊 Preparing data for RAGAS evaluation...")

dataset_dict = {
    'question': [r['question'] for r in results],
    'answer': [r['answer'] for r in results],
    'contexts': [r['contexts'] for r in results],
    'ground_truth': [r['ground_truth'] for r in results]
}

dataset = Dataset.from_dict(dataset_dict)
print("✅ Dataset prepared")

# Run RAGAS evaluation
print("\n🔬 Running RAGAS evaluation...")
print("⏳ This will take 1-2 minutes...\n")

evaluation_result = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=llm,
    embeddings=embeddings
)

# Display results
print("\n" + "="*70)
print("📊 EVALUATION RESULTS")
print("="*70)

# Extract scores from the result object
faithfulness_score = evaluation_result['faithfulness']
relevancy_score = evaluation_result['answer_relevancy']
precision_score = evaluation_result['context_precision']
recall_score = evaluation_result['context_recall']

print(f"""
Overall Scores:
───────────────────────────────────────────────────────

Faithfulness:        {faithfulness_score:.2f}
  → Measures if answers are factually grounded in source
  → Higher = Less hallucination
  → Target: > 0.90

Answer Relevancy:    {relevancy_score:.2f}
  → Measures if answers address the question
  → Higher = More on-topic
  → Target: > 0.85

Context Precision:   {precision_score:.2f}
  → Measures if retrieved chunks are useful
  → Higher = Better retrieval
  → Target: > 0.85

Context Recall:      {recall_score:.2f}
  → Measures if all needed info was retrieved
  → Higher = More complete
  → Target: > 0.85

───────────────────────────────────────────────────────
""")

# Save detailed results
df_results = pd.DataFrame(results)
df_results.to_csv('evaluation_results.csv', index=False)
print("💾 Detailed results saved to: evaluation_results.csv")

# Summary
print("\n" + "="*70)
print("📈 SUMMARY")
print("="*70)

avg_score = (
    faithfulness_score + 
    relevancy_score + 
    precision_score + 
    recall_score
) / 4

print(f"""
Average Score: {avg_score:.2f}

Interpretation:
  0.90 - 1.00  →  Excellent (Production-ready) ✅
  0.80 - 0.90  →  Good (Minor improvements needed) ⚠️
  0.70 - 0.80  →  Fair (Needs optimization) 🔧
  < 0.70       →  Poor (Significant issues) ❌

System Status: {'✅ PRODUCTION-READY!' if avg_score >= 0.90 else '⚠️ NEEDS TUNING' if avg_score >= 0.80 else '🔧 REQUIRES OPTIMIZATION'}

""")

print("="*70)
print("✅ EVALUATION COMPLETE!")
print("="*70)