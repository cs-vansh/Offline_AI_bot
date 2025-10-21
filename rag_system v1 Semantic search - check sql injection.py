#!/usr/bin/env python3
"""
RAG System: txtai + Ollama Integration
Combines txtai embeddings for retrieval with Ollama LLM for answer generation
"""

import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizers parallelism to avoid fork warnings

import json
import subprocess
from txtai.embeddings import Embeddings
import build_index  # Your existing build_index.py for data extraction

class RAGSystem:
    def __init__(self, index_path=None, ollama_model="llama3.2:3b"):
        """Initialize RAG system with txtai embeddings and Ollama"""
        # Use absolute path relative to executable location if no path provided
        if index_path is None:
            if getattr(sys, 'frozen', False):
                # Running as PyInstaller bundle
                base_dir = os.path.dirname(sys.executable)
            else:
                # Running as script
                base_dir = os.path.dirname(__file__)
            index_path = os.path.join(base_dir, "index")
        
        self.index_path = index_path
        self.ollama_model = ollama_model
        self.embeddings = None
        self.content_map = {}
        self.ollama_path = self._find_ollama_path()  # Find Ollama executable
        
        print(f"🤖 Initializing RAG System...")
        print(f"   - txtai index: {index_path}")
        print(f"   - Ollama model: {ollama_model}")
        print(f"   - Ollama path: {self.ollama_path}")
        
        self._load_embeddings()
        self._check_ollama()
    
    def _load_embeddings(self):
        """Load txtai embeddings"""
        try:
            self.embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
            self.embeddings.load(self.index_path)
            print("✅ txtai embeddings loaded successfully")
            
            # Check if content is available in the index
            try:
                test_result = self.embeddings.search("SELECT text FROM txtai LIMIT 1")
                if test_result and isinstance(test_result[0], dict) and 'text' in test_result[0]:
                    self.has_content = True
                    print("✅ Index has content storage - can retrieve text directly!")
                else:
                    self.has_content = False
                    print("⚠️ Index doesn't have content storage - will need Excel files for content")
            except:
                self.has_content = False
                print("⚠️ Index doesn't support SQL queries - will need Excel files for content")
                
        except Exception as e:
            print(f"❌ Failed to load embeddings: {e}")
            raise
    
    def _find_ollama_path(self):
        """Find the Ollama executable path"""
        possible_paths = [
            '/usr/local/bin/ollama',  # Common brew install location
            '/opt/homebrew/bin/ollama',  # Apple Silicon brew location
            '/usr/bin/ollama',  # System install
            'ollama'  # Fallback to PATH
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        return 'ollama'  # Fallback
    
    def _check_ollama(self):
        """Check if Ollama and the model are available"""
        try:
            # Check if Ollama is running
            result = subprocess.run([self.ollama_path, 'list'], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Ollama is not running. Please start it with: ollama serve")
                return False
            
            # Check if our model is available
            if self.ollama_model not in result.stdout:
                print(f"⚠️ Model {self.ollama_model} not found. Installing...")
                install_result = subprocess.run([self.ollama_path, 'pull', self.ollama_model], 
                                              capture_output=True, text=True)
                if install_result.returncode == 0:
                    print(f"✅ Model {self.ollama_model} installed successfully")
                else:
                    print(f"❌ Failed to install model: {install_result.stderr}")
                    return False
            else:
                print(f"✅ Ollama model {self.ollama_model} is available")
            
            return True
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama first.")
            return False
    
    def load_content_from_excel(self, excel_files):
        """Load actual content from Excel files to map document IDs to text"""
        print(f"📁 Loading content from {len(excel_files)} Excel files...")
        
        self.content_map = {}
        doc_id = 0
        
        for file_path in excel_files:
            if not os.path.exists(file_path):
                print(f"⚠️ File not found: {file_path}")
                continue
            
            try:
                entries = build_index.extract_qa_from_excel(file_path)
                for entry in entries:
                    self.content_map[doc_id] = {
                        'content': entry,
                        'source': os.path.basename(file_path)
                    }
                    doc_id += 1
                print(f"✅ Loaded {len(entries)} entries from {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        print(f"📊 Total content entries loaded: {len(self.content_map)}")
    
    def _escape_sql_string(self, text):
        """Ultra-conservative SQL string cleaning for txtai compatibility"""
        if not isinstance(text, str):
            return str(text)
        
        # Start with the original text
        escaped = text
        
        # 1. Remove ALL potentially problematic characters
        escaped = escaped.replace("'", "")  # Remove single quotes
        escaped = escaped.replace('"', "")  # Remove double quotes
        escaped = escaped.replace("`", "")  # Remove backticks
        escaped = escaped.replace("\\", "") # Remove backslashes
        escaped = escaped.replace(";", "")  # Remove semicolons
        escaped = escaped.replace("--", "") # Remove SQL comments
        escaped = escaped.replace("/*", "") # Remove block comments
        escaped = escaped.replace("*/", "") # Remove block comments
        escaped = escaped.replace("(", "")  # Remove parentheses
        escaped = escaped.replace(")", "")  # Remove parentheses
        escaped = escaped.replace("[", "")  # Remove brackets
        escaped = escaped.replace("]", "")  # Remove brackets
        escaped = escaped.replace("{", "")  # Remove braces
        escaped = escaped.replace("}", "")  # Remove braces
        
        # 2. Remove dangerous SQL keywords entirely
        dangerous_keywords = [
            'UNION', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE',
            'ALTER', 'EXEC', 'EXECUTE', 'SCRIPT', 'TRUNCATE', 'GRANT', 'REVOKE'
        ]
        
        for keyword in dangerous_keywords:
            import re
            # Remove keyword regardless of word boundaries (more aggressive)
            escaped = re.sub(re.escape(keyword), '', escaped, flags=re.IGNORECASE)
        
        # 3. Clean up multiple spaces and normalize
        escaped = ' '.join(escaped.split())  # Remove extra whitespace
        
        # 4. Limit length
        max_length = 500  # Shorter limit for safety
        if len(escaped) > max_length:
            escaped = escaped[:max_length]
        
        # 5. Only keep alphanumeric, spaces, and basic punctuation
        import re
        escaped = re.sub(r'[^a-zA-Z0-9\s\.\,\?\!]', '', escaped)
        
        # 6. Final cleanup
        escaped = escaped.strip()
        
        return escaped

    def retrieve_documents(self, question, num_docs=3):
        """Retrieve relevant documents using txtai - with smart content retrieval"""
        try:
            if self.has_content:
                # Method 1: Try SQL query with comprehensive escaping (most effective for content retrieval)
                try:
                    escaped_question = self._escape_sql_string(question)
                    
                    if not escaped_question or len(escaped_question.strip()) < 2:
                        print("⚠️ Question too short after escaping, using basic search")
                        results = self.embeddings.search(question, num_docs)
                        return self._process_basic_search_results(results)
                    
                    # Use SQL query to get content directly from index (like sql_qa.py)
                    query = f"select text, score from txtai where similar('{escaped_question}') limit {num_docs}"
                    results = self.embeddings.search(query)
                    
                    retrieved_docs = []
                    for i, result in enumerate(results):
                        if isinstance(result, dict):
                            retrieved_docs.append({
                                'doc_id': result.get('id', f'sql_{i}'),
                                'score': result.get('score', 0.0),
                                'content': result.get('text', 'No content'),
                                'source': 'Index (SQL)'
                            })
                    
                    if retrieved_docs:
                        return retrieved_docs
                    else:
                        print("⚠️ SQL query returned no results, trying fallback methods")
                    
                except Exception as sql_error:
                    print(f"⚠️ SQL query failed: {sql_error}")
                    # Continue to fallback methods
                
                # Method 2: Basic search + individual content queries (fallback)
                try:
                    results = self.embeddings.search(question, num_docs)
                    
                    retrieved_docs = []
                    for result in results:
                        if isinstance(result, (list, tuple)) and len(result) >= 2:
                            doc_id, score = result[0], result[1]
                            
                            # Try to get content for this specific document ID
                            try:
                                content_query = f"select text from txtai where id = {doc_id}"
                                content_result = self.embeddings.search(content_query)
                                
                                if content_result and isinstance(content_result[0], dict):
                                    content = content_result[0].get('text', f'Document {doc_id}')
                                else:
                                    content = f'Document {doc_id} - content not available'
                                    
                            except Exception as content_error:
                                print(f"⚠️ Failed to get content for doc {doc_id}: {content_error}")
                                content = f'Document {doc_id} - content retrieval failed'
                            
                            retrieved_docs.append({
                                'doc_id': doc_id,
                                'score': score,
                                'content': content,
                                'source': 'Index (ID-based)'
                            })
                        elif isinstance(result, dict):
                            # Result already contains content
                            retrieved_docs.append({
                                'doc_id': result.get('id', f'dict_{len(retrieved_docs)}'),
                                'score': result.get('score', 0.0),
                                'content': result.get('text', 'No content'),
                                'source': 'Index (Direct)'
                            })
                    
                    if retrieved_docs:
                        return retrieved_docs
                    
                except Exception as search_error:
                    print(f"⚠️ Basic search failed: {search_error}")
                
                # Method 3: Final fallback to Excel mapping if available
                print("⚠️ All content retrieval methods failed, using Excel mapping if available")
                results = self.embeddings.search(question, num_docs)
                return self._process_basic_search_results(results)
                
            else:
                # No content in index - use Excel mapping
                results = self.embeddings.search(question, num_docs)
                return self._process_basic_search_results(results)
                
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            # Ultimate fallback
            try:
                results = self.embeddings.search(question, num_docs)
                return self._process_basic_search_results(results)
            except:
                return []

    def _process_basic_search_results(self, results):
        """Helper method to process basic search results"""
        retrieved_docs = []
        for result in results:
            if isinstance(result, (list, tuple)) and len(result) >= 2:
                doc_id, score = result[0], result[1]
            elif isinstance(result, dict):
                doc_id = result.get('id', result.get('doc_id', 0))
                score = result.get('score', 0.0)
            else:
                continue
                
            if doc_id in self.content_map:
                retrieved_docs.append({
                    'doc_id': doc_id,
                    'score': score,
                    'content': self.content_map[doc_id]['content'],
                    'source': self.content_map[doc_id]['source']
                })
            else:
                retrieved_docs.append({
                    'doc_id': doc_id,
                    'score': score,
                    'content': f"[Document {doc_id} - content not available]",
                    'source': 'Unknown'
                })
        
        return retrieved_docs
    
    def generate_answer(self, question, retrieved_docs):
        """Generate answer using Ollama based on retrieved documents"""
        if not retrieved_docs:
            return "❌ No relevant documents found."
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            score_info = f"(Relevance: {doc['score']:.3f})"
            context_parts.append(f"Document {i} {score_info}:\n{doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Create prompt for the LLM - enhanced for security assessments
        prompt = f"""
You are an information security consultant responding on behalf of Leegality (Grey Swift Pvt. Ltd.) to a client security assessment. Provide accurate, diplomatic, and compliant answers.

Important rules:
- Answer only the specific 1 question (may have subparts - but will be considered 1 single question) under "QUESTION(s) TO ANSWER" till before the start of “REFERENCE DOCUMENTS” section.
- Other content under “REFERENCE DOCUMENTS” is for help and contains relevant answers.
- “Guidelines” section will give the instructions for answer framing and rules to follow at all times.

Context:
- You represent Grey Swift or Leegality (third party under review).
- Client is performing due diligence on security.
- Responses are used for compliance/risk assessment.
- Maintain professional tone with transparency.

QUESTION(s) TO ANSWER:
{question}

REFERENCE DOCUMENTS:
{context}

Guidelines:

1. Accuracy & Use of Context
- Use only the information in the provided reference documents.
- Do not assume or add anything beyond the given context.
- Do not mention or reference metadata such as document scores, document numbers, or system labels (e.g., "Document 1", "Score 0.789").
- Reference specific policies, controls, or procedures where available.
- If the context contains links, include them in the response (strictly as given in the context).

2. Diplomacy
- For strengths: be specific and confident.
- For gaps: do not admit directly; provide vague but acceptable wording.
- For unclear areas: avoid addressing unless strictly necessary; only use “based on available information” if unavoidable.

3. Focus
- Answer only the given question(s).
- Do not restate the question, number items, or add commentary/opinions.
- Never use phrases like “based on my knowledge” or “based on the context given”.

4. Response Structure
- Begin with a direct answer (Yes/No/Partial), only if relevant.
- Provide a brief explanation of the current state.
- Add specific details of controls, practices, or processes that support the answer.
- Use generic wording such as “the organization’s data” rather than client names, unless required by the context.
- Do not justify clear No/Not Applicable answers; justify only when a Yes is expected.

5. Compliance
- Frame answers around security objectives and compliance posture.
- Reference standards/frameworks if they are mentioned in the context.

6. Format
- Use plain text only (no markdown, bold, or headings).
- Write in full sentences and short paragraphs, not lists.
- Do not include the question itself in the response.

7. Evidence
- Add an "Evidence:" section only if explicit document titles, filenames, or policy names are present in the provided context.
- Valid examples: "Information Security Policy_2025.pdf", "Access Control Procedure v1.03.pdf". Generally with .pdf in the name or some hyperlinks.
- Invalid examples: "Reference Document 1", "Data Centre Information", or any descriptive text that is not a real filename/title.
- If multiple valid versions exist, list only the latest version (based on year, version, or naming convention).
- If no explicit document names are available, omit the Evidence section entirely.

RESPONSE FORMAT:
Provide a professional, structured, focused response that directly addresses the question(s) above while representing Leegality's security posture accurately.

RESPONSE:"""
        
        try:
            # Call Ollama
            result = subprocess.run([
                self.ollama_path, 'run', self.ollama_model
            ], input=prompt, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"❌ LLM generation failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "❌ LLM response timed out"
        except Exception as e:
            return f"❌ Error generating answer: {e}"
    
    def ask(self, question, num_docs=3, show_sources=True):
        """Complete RAG pipeline: retrieve + generate"""
        print(f"🔍 Question: {question}")
        print("=" * 60)
        
        # Step 1: Retrieve relevant documents
        print("📊 Retrieving relevant documents...")
        retrieved_docs = self.retrieve_documents(question, num_docs)
        
        if not retrieved_docs:
            return "❌ No relevant documents found."
        
        # Show retrieval results
        if show_sources:
            print("\n📄 Retrieved Documents:")
            for i, doc in enumerate(retrieved_docs, 1):
                score_quality = "🟢" if doc['score'] > 0.7 else "🟡" if doc['score'] > 0.5 else "🟠"
                print(f"   {i}. {score_quality} Doc #{doc['doc_id']} (Score: {doc['score']:.3f})")
                content_preview = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
                print(f"      {content_preview}")
        
        # Step 2: Generate answer using LLM
        print(f"\n🤖 Generating answer with {self.ollama_model}...")
        answer = self.generate_answer(question, retrieved_docs)
        
        print(f"\n💡 Answer:")
        print("-" * 40)
        print(answer)
        
        if show_sources:
            print(f"\n📚 Sources:")
            sources = set(doc['source'] for doc in retrieved_docs if doc['source'] != 'Unknown')
            for source in sources:
                print(f"   - {source}")
        
        return answer

def main():
    """Interactive RAG system"""
    print("🚀 RAG System: txtai + Ollama")
    print("=" * 50)
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Only ask for Excel files if index doesn't have content
    if not rag.has_content:
        print("\n📁 Excel Files Setup (for content mapping):")
        print("Your index doesn't have content storage. You can:")
        print("1. Provide Excel files for content mapping")
        print("2. Rebuild index with content storage using build_index.py")
        print("\nEnter paths to your Excel files (one per line, empty line to finish):")
        
        excel_files = []
        while True:
            path = input("Excel file path: ").strip()
            if not path:
                break
            if os.path.exists(path):
                excel_files.append(path)
                print(f"✅ Added: {os.path.basename(path)}")
            else:
                print(f"❌ File not found: {path}")
        
        if excel_files:
            rag.load_content_from_excel(excel_files)
        else:
            print("⚠️ No Excel files provided. Will work with document IDs only.")
    else:
        print("✅ Index has content storage - no Excel files needed!")
    
    # Interactive Q&A
    print(f"\n🤖 RAG System Ready!")
    print("Ask questions (type 'quit' to exit):")
    
    while True:
        question = input("\n❓ Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question:
            try:
                rag.ask(question)
            except Exception as e:
                print(f"❌ Error: {e}")
        
    print("\n👋 Thanks for using the RAG system!")

if __name__ == "__main__":
    main()
