#!/usr/bin/env python3
"""
Quick RAG Test - Works with your existing setup
"""

# Set environment variables BEFORE any imports to fix OpenMP conflicts
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import subprocess
from txtai.embeddings import Embeddings
import build_index

def quick_rag_test(question="What is the data policy?"):
    """Quick test of RAG system with your existing index"""
    print(f"🔍 Testing RAG with question: '{question}'")
    print("=" * 60)
    
    # Step 1: Load embeddings and search
    try:
        embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
        embeddings.load("index")
        print("✅ Loaded txtai index")
        
        # Search for relevant documents
        results = embeddings.search(question, 5)
        print(f"\n📊 Found {len(results)} relevant documents:")
        
        if not results:
            print("❌ No results found - index might be empty or corrupted")
            return
        
        # Debug: Check the format of results
        print(f"Debug: First result format: {results[0] if results else 'No results'}")
        
        for i, result in enumerate(results, 1):
            try:
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    doc_id, score = result[0], result[1]
                elif isinstance(result, dict):
                    doc_id = result.get('id', result.get('doc_id', i))
                    score = result.get('score', 0.0)
                else:
                    print(f"⚠️ Unexpected result format: {result}")
                    continue
                    
                quality = "🟢" if score > 0.7 else "🟡" if score > 0.5 else "🟠"
                print(f"   {i}. {quality} Doc #{doc_id} (Score: {score:.3f})")
            except Exception as e:
                print(f"⚠️ Error processing result {i}: {e}")
                print(f"   Raw result: {result}")
        
        # Step 2: Try to get actual content (if Excel files available)
        excel_files = []
        # Check if Excel files are in done/ directory
        done_dir = "done"
        if os.path.exists(done_dir):
            for file in os.listdir(done_dir):
                if file.endswith('.xlsx'):
                    excel_files.append(os.path.join(done_dir, file))
        
        content_map = {}
        if excel_files:
            print(f"\n📁 Loading content from {len(excel_files)} Excel files...")
            doc_id = 0
            for file_path in excel_files:
                try:
                    entries = build_index.extract_qa_from_excel(file_path)
                    for entry in entries:
                        content_map[doc_id] = entry
                        doc_id += 1
                except Exception as e:
                    print(f"⚠️ Error loading {file_path}: {e}")
            
            print(f"✅ Loaded content for {len(content_map)} documents")
        
        # Step 3: Show relevant content
        print(f"\n📄 Relevant Content:")
        print("-" * 40)
        
        context_for_llm = []
        for i, result in enumerate(results[:5], 1):
            try:
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    doc_id, score = result[0], result[1]
                elif isinstance(result, dict):
                    doc_id = result.get('id', result.get('doc_id', i))
                    score = result.get('score', 0.0)
                else:
                    print(f"⚠️ Unexpected result format in content processing: {result}")
                    continue
                    
                if doc_id in content_map:
                    content = content_map[doc_id]
                    print(f"\n{i}. Document {doc_id} (Score: {score:.3f}):")
                    preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"   {preview}")
                    context_for_llm.append(content)
                else:
                    print(f"\n{i}. Document {doc_id} (Score: {score:.3f}):")
                    print(f"   [Content not available - document ID only]")
            except Exception as e:
                print(f"⚠️ Error processing result {i}: {e}")
        
        # Step 4: Try Ollama if available and context exists
        if context_for_llm:
            print(f"\n🤖 Attempting to generate answer with Ollama...")
            
            context = "\n\n".join(context_for_llm)
            prompt = f"""Based on the following context, please answer the question concisely:

Question: {question}

Context:
{context}

Answer:"""
            
            try:
                result = subprocess.run([
                    'ollama', 'run', 'llama3.2:3b'
                ], input=prompt, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print(f"\n💡 Generated Answer:")
                    print("-" * 30)
                    print(result.stdout.strip())
                else:
                    print(f"⚠️ Ollama not ready yet or model not available")
                    print(f"Raw context available for manual review above")
                    
            except Exception as e:
                print(f"⚠️ Ollama error: {e}")
                print(f"You can use the context above to answer manually")
        else:
            print(f"\n💡 To get actual content:")
            print(f"   1. Make sure your Excel files are accessible")
            print(f"   2. Or rebuild index with content storage")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def test_multiple_questions():
    """Test with multiple sample questions"""
    questions = [
        "What is the data policy?",
        "How is production access managed?",
        "What are the security requirements?",
        "Describe the vendor evaluation process",
        "What is the backup policy?"
    ]
    
    for question in questions:
        quick_rag_test(question)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Use question from command line
        question = " ".join(sys.argv[1:])
        quick_rag_test(question)
    else:
        # Interactive mode
        print("🤖 Quick RAG Tester")
        print("=" * 30)
        print("Choose an option:")
        print("1. Test with sample questions")
        print("2. Ask a custom question")
        
        choice = input("\nChoice (1/2): ").strip()
        
        if choice == "1":
            test_multiple_questions()
        else:
            question = input("Enter your question: ").strip()
            if question:
                quick_rag_test(question)
