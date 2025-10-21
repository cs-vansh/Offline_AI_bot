#!/usr/bin/env python3
"""
SQL-based Question Answering with txtai
Uses SQL queries to get actual content, not just IDs
"""

# Set environment variables BEFORE any imports to fix OpenMP conflicts
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from txtai.embeddings import Embeddings

class SQLQuestionAnswerer:
    def __init__(self, index_path="index"):
        """Initialize with existing index"""
        self.embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
        self.embeddings.load(index_path)
        print(f"✅ Loaded index from {index_path}")
        
        # Check if content is available
        try:
            test_result = self.embeddings.search("SELECT text FROM txtai LIMIT 1")
            if test_result and 'text' in test_result[0]:
                self.has_content = True
                print("✅ Index has content storage - SQL queries will work!")
            else:
                self.has_content = False
                print("⚠️ Index doesn't have content storage - only IDs available")
        except:
            self.has_content = False
            print("⚠️ Index doesn't support SQL queries - rebuild needed")
    
    def question(self, text, limit=1):
        """Get answer using SQL query approach"""
        if not self.has_content:
            print("❌ Cannot get text content - index needs to be rebuilt with content storage")
            return self.embeddings.search(text, limit)
        
        try:
            # Your SQL approach - get text and score
            query = f"select text, score from txtai where similar('{text}') limit {limit}"
            return self.embeddings.search(query)
        except Exception as e:
            print(f"❌ SQL query failed: {e}")
            return []
    
    def get_best_answer(self, text):
        """Get the single best answer with formatted output"""
        results = self.question(text, 1)
        
        if not results:
            return "❌ No results found"
        
        if self.has_content:
            result = results[0]
            score = result.get('score', 0)
            content = result.get('text', 'No content')
            
            # Score interpretation
            if score > 0.7:
                quality = "🟢 EXCELLENT"
            elif score > 0.5:
                quality = "🟡 GOOD"
            elif score > 0.3:
                quality = "🟠 FAIR"
            else:
                quality = "🔴 WEAK"
            
            return f"{quality} (Score: {score:.3f})\n📄 {content}"
        else:
            return f"Document ID: {results[0][0]}, Score: {results[0][1]:.3f}"
    
    def search_multiple(self, text, limit=5):
        """Get multiple results"""
        if not self.has_content:
            return self.embeddings.search(text, limit)
        
        query = f"select text, score from txtai where similar('{text}') limit {limit}"
        return self.embeddings.search(query)
    
    def browse_content(self, limit=5):
        """Browse all content in the index"""
        if not self.has_content:
            print("❌ Cannot browse content - index needs content storage")
            return []
        
        return self.embeddings.search(f"SELECT text FROM txtai LIMIT {limit}")

def main():
    print("🤖 SQL-based txtai Question Answerer")
    print("=" * 50)
    
    qa = SQLQuestionAnswerer()
    
    if not qa.has_content:
        print("\n💡 To enable SQL queries with content:")
        print("1. Run: python rebuild_with_content.py")
        print("2. This will rebuild your index with content storage")
        print("3. Then you can use SQL queries to get actual text!")
        return
    
    print("\n🎉 Your index supports SQL queries! Try these:")
    
    # Test questions
    test_questions = [
        "What is the timezone of NYC?",
        "What is the data policy?",
        "How is production access managed?",
        "What is your data retention policy?"
    ]
    
    for question_text in test_questions:
        print(f"\n🔍 Question: '{question_text}'")
        print("-" * 50)
        answer = qa.get_best_answer(question_text)
        print(answer)
    
    # Interactive mode
    print("\n" + "="*50)
    print("💬 Interactive Mode - Enter your questions:")
    print("(Type 'quit' to exit)")
    
    while True:
        user_question = input("\n❓ Your question: ").strip()
        if user_question.lower() in ['quit', 'exit', 'q']:
            break
        if user_question:
            print(qa.get_best_answer(user_question))

if __name__ == "__main__":
    main()
