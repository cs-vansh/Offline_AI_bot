#!/usr/bin/env python3
"""
RAG Excel Processor GUI
Based on: rag_system v1 Semantic search - check sql injection.py
Processes Excel files row by row, using each row as a question and adds RAG responses
"""

# Set environment variables BEFORE any imports to fix OpenMP conflicts
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import json
import subprocess
import threading
import time
from txtai.embeddings import Embeddings
from pathlib import Path

class RAGExcelProcessor:
    def __init__(self, index_path=None, ollama_model="llama3.2:3b"):
        """Initialize RAG system with txtai embeddings and Ollama (based on v1 logic)"""
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
        self.has_content = False
        
        # Processing control variables
        self.is_paused = False
        self.should_stop = False
        self.continue_on_error = True
        self.error_count = 0
        self.max_consecutive_errors = 5
        
        # Setup temp directory
        self.temp_dir = self._setup_temp_directory()
        
        # Find Ollama executable
        self.ollama_path = self._find_ollama_path()
        
        self._load_embeddings()
        self._check_ollama()
    
    def _load_embeddings(self):
        """Load txtai embeddings (from v1 file logic)"""
        try:
            self.embeddings = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
            self.embeddings.load(self.index_path)
            print("✅ txtai embeddings loaded successfully")
            
            # Check if content is available in the index (from v1 logic)
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
            print(f"❌ Failed to load txtai embeddings: {e}")
            self.embeddings = None
    
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
        """Check if Ollama is available (from v1 file logic)"""
        try:
            result = subprocess.run([self.ollama_path, 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                models = result.stdout
                if self.ollama_model.split(':')[0] in models:
                    print(f"✅ Ollama model {self.ollama_model} is available")
                else:
                    print(f"⚠️ Model {self.ollama_model} not found. Available models:\n{models}")
            else:
                print(f"❌ Ollama not responding: {result.stderr}")
        except Exception as e:
            print(f"❌ Error checking Ollama: {e}")
    
    def _setup_temp_directory(self):
        """Create and return the temp directory path"""
        # Get user's Documents directory
        home_dir = Path.home()
        documents_dir = home_dir / "Documents"
        temp_dir = documents_dir / "TempFiles-OfflineAIBot"
        
        # Create directory if it doesn't exist
        try:
            temp_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Temp directory ready: {temp_dir}")
            return temp_dir
        except Exception as e:
            print(f"⚠️ Failed to create temp directory: {e}")
            # Fallback to current directory
            fallback_dir = Path("TempFiles-OfflineAIBot")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Using fallback temp directory: {fallback_dir}")
            return fallback_dir
    
    def _cleanup_old_temp_files(self, original_file_path, keep_last_n=2, sheet_name=None):
        """Clean up old temporary files, keeping only the last N versions"""
        try:
            # Find all temp files for this original file and sheet
            temp_files = self._find_temp_files(original_file_path, sheet_name)
            
            if len(temp_files) <= keep_last_n:
                return  # Nothing to clean up
            
            # Sort by row count (ascending order)
            temp_files.sort(key=lambda x: x['row_count'])
            
            # Files to delete (all except the last N)
            files_to_delete = temp_files[:-keep_last_n]
            
            deleted_count = 0
            for temp_file in files_to_delete:
                try:
                    file_path = Path(temp_file['path'])
                    if file_path.exists():
                        file_path.unlink()  # Delete the file
                        print(f"🗑️ Deleted old temp file: {temp_file['filename']}")
                        deleted_count += 1
                except Exception as delete_error:
                    print(f"⚠️ Failed to delete {temp_file['filename']}: {delete_error}")
            
            if deleted_count > 0:
                remaining_files = [tf['filename'] for tf in temp_files[-keep_last_n:]]
                print(f"✅ Cleaned up {deleted_count} old temp files. Kept: {', '.join(remaining_files)}")
                
        except Exception as e:
            print(f"⚠️ Error during temp file cleanup: {e}")
    
    def pause_processing(self):
        """Pause the current processing"""
        self.is_paused = True
        print("⏸️ Processing paused by user")
    
    def resume_processing(self):
        """Resume the paused processing"""
        self.is_paused = False
        print("▶️ Processing resumed by user")
    
    def stop_processing(self):
        """Stop the current processing completely"""
        self.should_stop = True
        self.is_paused = False
        print("⏹️ Processing stopped by user")
    
    def reset_processing_state(self):
        """Reset processing control variables"""
        self.is_paused = False
        self.should_stop = False
        self.error_count = 0
    
    def _find_temp_files(self, original_file_path, sheet_name=None):
        """Find all temporary files for the given original file and optionally specific sheet"""
        import glob
        
        # Get just the filename without path and extension
        original_filename = Path(original_file_path).stem
        
        # Look for temp files in the temp directory
        if sheet_name:
            # Clean sheet name for filename (replace problematic characters)
            safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_sheet_name = safe_sheet_name.replace(' ', '_')
            pattern = str(self.temp_dir / f"{original_filename}_sheet_{safe_sheet_name}_temp_progress_*.xlsx")
        else:
            # Backward compatibility - look for files without sheet name
            pattern = str(self.temp_dir / f"{original_filename}_temp_progress_*.xlsx")
        
        temp_files = glob.glob(pattern)
        
        temp_info = []
        for temp_file in temp_files:
            # Extract row count from filename
            try:
                filename = Path(temp_file).name
                if sheet_name:
                    row_count = int(filename.split('_temp_progress_')[1].replace('.xlsx', ''))
                else:
                    row_count = int(filename.split('_temp_progress_')[1].replace('.xlsx', ''))
                temp_info.append({
                    'path': temp_file,
                    'filename': filename,
                    'row_count': row_count,
                    'mtime': Path(temp_file).stat().st_mtime
                })
            except (ValueError, IndexError):
                continue
        
        return temp_info
    
    def _find_last_processed_row(self, df):
        """Find the last row that was successfully processed"""
        response_column = "RAG_Response"
        
        if response_column not in df.columns:
            return 0
        
        # Find the last row with a non-empty response
        for index in range(len(df) - 1, -1, -1):
            response = str(df.iloc[index][response_column]).strip()
            if response and response != "nan" and response != "":
                return index + 1  # Start from the next row
        
        return 0  # Start from the beginning if no processed rows found
    
    def _escape_sql_string(self, text):
        """Ultra-conservative SQL string cleaning (exact from v1 file)"""
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
        """Retrieve relevant documents using txtai (exact logic from v1 file)"""
        try:
            if self.has_content:
                # Method 1: Try SQL query with comprehensive escaping
                try:
                    escaped_question = self._escape_sql_string(question)
                    
                    if not escaped_question or len(escaped_question.strip()) < 2:
                        print("⚠️ Question too short after escaping, using basic search")
                        results = self.embeddings.search(question, num_docs)
                        return self._process_basic_search_results(results)
                    
                    # Use SQL query to get content directly from index
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
                
                # No more fallback methods available
                return []
            else:
                # No content in index - cannot retrieve documents
                return []
                
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []

    def generate_answer(self, question, retrieved_docs):
        """Generate answer using Ollama (exact logic from v1 file)"""
        if not retrieved_docs:
            return "❌ Error retrieving relevant documents - no content available in index."
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            # Just add content without document numbers or debug info
            context_parts.append(f"{doc['content']}\n")
        
        context = "\n---\n".join(context_parts)
        
        # Debug: Print detailed question and context being passed to LLM
        print("\n" + "="*100)
        print("🔍 DEBUG - COMPLETE QUESTION + CONTEXT BEING PASSED TO LLM")
        print("="*100)
        print(f"📝 QUESTION:\n{question}")
        print("\n" + "-"*80)
        print(f"📊 RETRIEVED {len(retrieved_docs)} DOCUMENTS:")
        
        # Show each document with metadata
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"\n📄 DOCUMENT {i}:")
            print(f"   Score: {doc.get('score', 'N/A')}")
            print(f"   Doc ID: {doc.get('doc_id', 'N/A')}")
            print(f"   Source: {doc.get('source', 'N/A')}")
            print(f"   Content Length: {len(str(doc.get('content', '')))} characters")
            print(f"   FULL Content:\n{str(doc.get('content', 'No content'))}")
            print("-" * 60)
        
        print("\n" + "-"*80)
        print(f"📚 FULL CONTEXT SENT TO LLM:")
        print("-" * 40)
        print(context)
        print("-" * 40)
        print(f"📏 Total Context Length: {len(context)} characters")
        print("="*100 + "\n")
        
        # Add custom prompt here
        prompt = f"""

You are an analyst responding on behalf of ...... to a client. Provide accurate, diplomatic, and compliant answers.

Important rules:
- Answer only the specific 1 question (may have subparts - but will be considered 1 single question) under "QUESTION(s) TO ANSWER" till before the start of “REFERENCE DOCUMENTS” section.
- Other content under “REFERENCE DOCUMENTS” is for help and contains relevant answers.
- “Guidelines” section will give the instructions for answer framing and rules to follow at all times.

Context:
- You represent ..........
- Responses are used for .........
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

5. Format
- Use plain text only (no markdown, bold, or headings).
- Write in full sentences and short paragraphs, not lists.
- Do not include the question itself in the response.

RESPONSE FORMAT:
Provide a professional, structured, focused response that directly addresses the question(s) above while representing the posture accurately.

Response:"""

        try:
            # Call Ollama (from v1 file logic)
            result = subprocess.run([
                self.ollama_path, 'run', self.ollama_model
            ], input=prompt, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                answer = result.stdout.strip()
                return answer if answer else "No response generated."
            else:
                return f"❌ LLM generation failed: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "❌ LLM response timed out"
        except Exception as e:
            return f"❌ Error generating answer: {e}"
    
    def get_rag_response(self, question):
        """Get complete RAG response for a question (retrieve + generate)"""
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retrieve_documents(question, num_docs=3)
            
            if not retrieved_docs:
                return "❌ No relevant documents found for your question."
            
            # Step 2: Generate answer using retrieved documents
            answer = self.generate_answer(question, retrieved_docs)
            
            return answer   
            
        except Exception as e:
            return f"❌ Error processing question: {e}"

    def process_excel_file(self, file_path, progress_callback=None, log_callback=None, 
                          continue_on_error=True, auto_save_interval=10, resume_from_temp=None):
        """Process Excel file with multi-sheet support - handles all sheets in workbook"""
        try:
            # Reset processing state
            self.reset_processing_state()
            self.continue_on_error = continue_on_error
            
            if log_callback:
                log_callback("🚀 Starting multi-sheet Excel processing...")
                log_callback(f"📁 File: {Path(file_path).name}")
                log_callback(f"🔍 RAG Model: {self.ollama_model}")
            
            # First, check what sheets are in the workbook
            try:
                excel_file = pd.ExcelFile(file_path)
                sheet_names = excel_file.sheet_names
                if log_callback:
                    log_callback(f"📊 Found {len(sheet_names)} sheets: {', '.join(sheet_names)}")
            except Exception as e:
                if log_callback:
                    log_callback(f"❌ Error reading Excel file: {e}")
                return None
            
            # Process each sheet
            all_output_paths = []
            overall_processed = 0
            overall_success = 0
            overall_errors = 0
            
            for sheet_idx, sheet_name in enumerate(sheet_names):
                if self.should_stop:
                    if log_callback:
                        log_callback("⏹️ Processing stopped by user")
                    break
                    
                if log_callback:
                    log_callback(f"\n📋 Processing Sheet {sheet_idx + 1}/{len(sheet_names)}: '{sheet_name}'")
                
                # Process individual sheet
                sheet_result = self._process_single_sheet(
                    file_path, sheet_name, sheet_idx, len(sheet_names),
                    progress_callback, log_callback, continue_on_error, auto_save_interval
                )
                
                if sheet_result:
                    all_output_paths.append(sheet_result['output_path'])
                    overall_processed += sheet_result['processed_count']
                    overall_success += sheet_result['success_count']
                    overall_errors += sheet_result['error_count']
            
            # Final summary
            if log_callback:
                log_callback(f"\n🎯 Multi-Sheet Processing Complete!")
                log_callback(f"   • Sheets processed: {len(all_output_paths)}/{len(sheet_names)}")
                log_callback(f"   • Total rows processed: {overall_processed}")
                log_callback(f"   • Total successful: {overall_success}")
                log_callback(f"   • Total errors: {overall_errors}")
                if overall_processed > 0:
                    log_callback(f"   • Overall success rate: {(overall_success/overall_processed*100):.1f}%")
                log_callback(f"   • Output files: {len(all_output_paths)}")
            
            # Return the list of all output files or the first one for compatibility
            return all_output_paths[0] if all_output_paths else None
            
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Error processing multi-sheet file: {e}")
            return None

    def _process_single_sheet(self, file_path, sheet_name, sheet_idx, total_sheets, 
                             progress_callback, log_callback, continue_on_error, auto_save_interval):
        """Process a single sheet within a workbook"""
        try:
            # Check for temp files for this specific sheet
            resume_from_temp = None
            temp_files = self._find_temp_files(file_path, sheet_name)
            
            if temp_files:
                latest_temp = max(temp_files, key=lambda x: x['row_count'])
                if log_callback:
                    log_callback(f"📂 Found temp file for sheet '{sheet_name}': {Path(latest_temp['path']).name}")
                    resume_from_temp = latest_temp['path']
            
            # Determine starting file and row for this sheet
            if resume_from_temp and os.path.exists(resume_from_temp):
                df = pd.read_excel(resume_from_temp)
                start_row = self._find_last_processed_row(df)
                if log_callback:
                    log_callback(f"📂 Resuming from temporary file: {Path(resume_from_temp).name}")
                    log_callback(f"▶️ Starting from row {start_row + 1}")
            else:
                # Read specific sheet from original Excel file
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                start_row = 0
                if log_callback:
                    log_callback(f"📁 Processing sheet: '{sheet_name}'")
            
            total_rows = len(df)
            
            if log_callback:
                log_callback(f"📊 Total rows in '{sheet_name}': {total_rows}")
                log_callback(f"📋 Columns: {list(df.columns)}")
                log_callback(f"⚙️ Continue on error: {continue_on_error}")
                log_callback(f"💾 Auto-save every {auto_save_interval} rows")
            
            # Create new column for RAG responses if it doesn't exist
            response_column = f"RAG_Response"
            if response_column not in df.columns:
                df[response_column] = ""
            
            # Track processing statistics
            processed_count = start_row  # Start counting from resume point
            success_count = 0
            error_count = 0
            consecutive_errors = 0
            
            # Process each row starting from the resume point
            for index in range(start_row, total_rows):
                row = df.iloc[index]
                # Check if processing should stop
                if self.should_stop:
                    if log_callback:
                        log_callback("⏹️ Processing stopped by user")
                    break
                
                try:
                    if log_callback:
                        log_callback(f"\n🔄 Processing row {index + 1}/{total_rows} in sheet '{sheet_name}'")
                    
                    # Combine all values in the row as the question
                    row_values = []
                    for col in df.columns:
                        if col != response_column:  # Skip the response column
                            cell_value = str(row[col]).strip()
                            if cell_value and cell_value.lower() != "nan":
                                row_values.append(cell_value)
                    
                    if not row_values:
                        df.at[index, response_column] = "❌ Empty row"
                        if log_callback:
                            log_callback("⚠️ Skipping empty row")
                        continue
                    
                    # Create question from row data
                    question = " | ".join(row_values)
                    
                    if log_callback:
                        log_callback(f"❓ Question: {question[:100]}...")
                    
                    # Retrieve documents with error handling
                    try:
                        docs = self.retrieve_documents(question, num_docs=3)
                        
                        if log_callback:
                            log_callback(f"📚 Found {len(docs)} relevant documents")
                    except Exception as retrieval_error:
                        error_msg = f"❌ Error retrieving documents: {str(retrieval_error)[:100]}"
                        if log_callback:
                            log_callback(error_msg)
                        
                        if not continue_on_error:
                            raise retrieval_error
                        
                        df.at[index, response_column] = error_msg
                        error_count += 1
                        consecutive_errors += 1
                        continue
                    
                    # Generate answer with error handling
                    try:
                        answer = self.generate_answer(question, docs)
                        
                        # Check if answer indicates an error
                        if answer.startswith("❌"):
                            error_count += 1
                            consecutive_errors += 1
                            if log_callback:
                                log_callback(f"⚠️ Error in answer generation: {answer[:100]}...")
                        else:
                            success_count += 1
                            consecutive_errors = 0  # Reset consecutive error count
                            if log_callback:
                                log_callback(f"✅ Response: {answer[:100]}...")
                        
                        # Store answer in Excel
                        df.at[index, response_column] = answer
                        
                    except Exception as generation_error:
                        error_msg = f"❌ Error generating answer: {str(generation_error)[:100]}"
                        if log_callback:
                            log_callback(error_msg)
                        
                        if not continue_on_error:
                            raise generation_error
                        
                        df.at[index, response_column] = error_msg
                        error_count += 1
                        consecutive_errors += 1
                    
                    processed_count += 1
                    
                    # Check for too many consecutive errors
                    if consecutive_errors >= self.max_consecutive_errors:
                        error_msg = f"🚨 Too many consecutive errors ({consecutive_errors}). Stopping sheet processing."
                        if log_callback:
                            log_callback(error_msg)
                        if not continue_on_error:
                            raise Exception(error_msg)
                        else:
                            if log_callback:
                                log_callback("⚠️ Continue-on-error enabled, attempting to continue...")
                            consecutive_errors = 0  # Reset and try to continue
                    
                    # Auto-save progress (with sheet-specific filename)
                    if processed_count % auto_save_interval == 0:
                        try:
                            # Save to temp directory with sheet-specific filename
                            original_filename = Path(file_path).stem
                            safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                            safe_sheet_name = safe_sheet_name.replace(' ', '_')
                            temp_filename = f"{original_filename}_sheet_{safe_sheet_name}_temp_progress_{processed_count}.xlsx"
                            temp_output_path = self.temp_dir / temp_filename
                            df.to_excel(temp_output_path, index=False)
                            if log_callback:
                                log_callback(f"💾 Auto-saved progress for sheet '{sheet_name}': {temp_output_path.name}")
                            
                            # Clean up old temp files for this sheet (keep only last 2 versions)
                            self._cleanup_old_temp_files(file_path, keep_last_n=2, sheet_name=sheet_name)
                            
                        except Exception as save_error:
                            if log_callback:
                                log_callback(f"⚠️ Auto-save failed: {save_error}")
                    
                    # Update progress (account for multiple sheets)
                    if progress_callback:
                        # Calculate overall progress across all sheets
                        sheet_progress = (index + 1) / total_rows
                        overall_progress = (sheet_idx + sheet_progress) / total_sheets
                        progress_callback(overall_progress * 100)
                
                except Exception as row_error:
                    error_msg = f"❌ Critical error processing row {index + 1} in sheet '{sheet_name}': {str(row_error)[:100]}"
                    if log_callback:
                        log_callback(error_msg)
                    
                    if not continue_on_error:
                        raise row_error
                    
                    df.at[index, response_column] = error_msg
                    error_count += 1
                    consecutive_errors += 1
            
            # Sheet statistics
            if log_callback:
                log_callback(f"\n📊 Sheet '{sheet_name}' Summary:")
                log_callback(f"   • Processed: {processed_count}/{total_rows}")
                log_callback(f"   • Successful: {success_count}")
                log_callback(f"   • Errors: {error_count}")
                log_callback(f"   • Success rate: {(success_count/processed_count*100):.1f}%" if processed_count > 0 else "N/A")
            
            # Save processed sheet
            original_filename = Path(file_path).stem
            safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_sheet_name = safe_sheet_name.replace(' ', '_')
            output_path = file_path.replace('.xlsx', f'_sheet_{safe_sheet_name}_with_rag_responses.xlsx')
            df.to_excel(output_path, index=False)
            
            if log_callback:
                log_callback(f"💾 Saved processed sheet: {Path(output_path).name}")
            
            return {
                'output_path': output_path,
                'processed_count': processed_count,
                'success_count': success_count,
                'error_count': error_count,
                'sheet_name': sheet_name
            }
            
        except Exception as e:
            if log_callback:
                log_callback(f"❌ Error processing sheet '{sheet_name}': {e}")
            return None


class RAGExcelGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RAG Excel Processor - Based on v1 Logic")
        self.root.geometry("800x600")
        
        # Initialize RAG system
        self.rag_system = None
        self.processing = False
        
        self.setup_ui()
        self.initialize_rag()
    
    def setup_ui(self):
        """Setup the GUI interface with tabs for Excel processing and Q&A"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="RAG Excel Processor", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Create tabs
        self.excel_tab = ttk.Frame(self.notebook)
        self.qa_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.excel_tab, text="Excel Processing")
        self.notebook.add(self.qa_tab, text="Ask Questions")
        
        # Setup Excel processing tab
        self.setup_excel_tab()
        
        # Setup Q&A tab
        self.setup_qa_tab()
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
    
    def setup_excel_tab(self):
        """Setup the Excel processing tab"""
        tab_frame = self.excel_tab
        
        tab_frame = self.excel_tab
        tab_frame.grid_columnconfigure(1, weight=1)
        tab_frame.grid_rowconfigure(7, weight=1)
        
        # File selection
        ttk.Label(tab_frame, text="Select Excel File:").grid(row=0, column=0, sticky=tk.W, padx=(10, 10), pady=(10, 10))
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(tab_frame, textvariable=self.file_path_var, width=50)
        self.file_entry.grid(row=0, column=1, padx=(10, 10), pady=(10, 10), sticky=(tk.W, tk.E))
        
        self.browse_button = ttk.Button(tab_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=(10, 10), pady=(10, 10))
        
        # Process button
        self.process_button = ttk.Button(tab_frame, text="Start Processing", 
                                       command=self.start_processing)
        self.process_button.grid(row=1, column=0, pady=(20, 10), padx=(10, 10))
        
        # Control buttons frame
        control_frame = ttk.Frame(tab_frame)
        control_frame.grid(row=1, column=1, columnspan=2, pady=(20, 10), sticky=tk.E, padx=(10, 10))
        
        self.pause_button = ttk.Button(control_frame, text="Pause", 
                                     command=self.pause_processing, state="disabled")
        self.pause_button.grid(row=0, column=0, padx=(5, 5))
        
        self.resume_button = ttk.Button(control_frame, text="Resume", 
                                      command=self.resume_processing, state="disabled")
        self.resume_button.grid(row=0, column=1, padx=(5, 5))
        
        self.stop_button = ttk.Button(control_frame, text="Stop", 
                                    command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=(5, 5))
        
        # Resume from temp button
        self.resume_temp_button = ttk.Button(control_frame, text="Resume from Temp", 
                                           command=self.resume_from_temp)
        self.resume_temp_button.grid(row=0, column=3, padx=(5, 5))
        
        # Options frame
        options_frame = ttk.LabelFrame(tab_frame, text="Processing Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 10), padx=(10, 10))
        
        # Continue on error checkbox
        self.continue_on_error_var = tk.BooleanVar(value=True)
        self.continue_on_error_cb = ttk.Checkbutton(options_frame, 
                                                  text="Continue processing on errors",
                                                  variable=self.continue_on_error_var)
        self.continue_on_error_cb.grid(row=0, column=0, sticky=tk.W)
        
        # Auto-save interval
        ttk.Label(options_frame, text="Auto-save every:").grid(row=0, column=1, padx=(20, 5), sticky=tk.W)
        self.auto_save_var = tk.StringVar(value="10")
        auto_save_spinbox = ttk.Spinbox(options_frame, from_=1, to=100, width=5, 
                                       textvariable=self.auto_save_var)
        auto_save_spinbox.grid(row=0, column=2, padx=(0, 5))
        ttk.Label(options_frame, text="rows").grid(row=0, column=3, sticky=tk.W)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(tab_frame, variable=self.progress_var, 
                                          maximum=100)
        self.progress_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 10), padx=(10, 10))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(tab_frame, textvariable=self.status_var)
        self.status_label.grid(row=4, column=0, columnspan=3, padx=(10, 10))
        
        # Log area
        ttk.Label(tab_frame, text="Processing Log:").grid(row=5, column=0, sticky=tk.W, pady=(20, 5), padx=(10, 10))
        
        self.log_text = scrolledtext.ScrolledText(tab_frame, width=80, height=15)
        self.log_text.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10), padx=(10, 10))
    
    def setup_qa_tab(self):
        """Setup the Q&A tab"""
        tab_frame = self.qa_tab
        tab_frame.grid_columnconfigure(1, weight=1)
        tab_frame.grid_rowconfigure(3, weight=1)
        
        # Status indicator
        self.qa_status_var = tk.StringVar(value="Ready for questions")
        self.qa_status_label = ttk.Label(tab_frame, textvariable=self.qa_status_var, 
                                       font=("Arial", 10, "italic"))
        self.qa_status_label.grid(row=0, column=0, columnspan=3, pady=(10, 10), padx=(10, 10))
        
        # Question input
        ttk.Label(tab_frame, text="Ask a Question:").grid(row=1, column=0, sticky=tk.W, padx=(10, 10), pady=(5, 5))
        
        self.question_text = scrolledtext.ScrolledText(tab_frame, width=60, height=3, wrap=tk.WORD)
        self.question_text.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=(10, 10), pady=(5, 5))
        
        # Ask button and Copy button frame
        button_frame = ttk.Frame(tab_frame)
        button_frame.grid(row=2, column=2, padx=(10, 10), pady=(5, 5))
        
        self.ask_button = ttk.Button(button_frame, text="Ask Question", 
                                   command=self.ask_question)
        self.ask_button.pack(pady=(0, 5))
        
        self.copy_button = ttk.Button(button_frame, text="Copy Answer", 
                                    command=self.copy_answer, state="disabled")
        self.copy_button.pack()
        
        # Answer area
        ttk.Label(tab_frame, text="Answer:").grid(row=3, column=0, sticky=(tk.W, tk.N), padx=(10, 10), pady=(10, 5))
        
        self.answer_text = scrolledtext.ScrolledText(tab_frame, width=80, height=20, wrap=tk.WORD)
        self.answer_text.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 10), pady=(0, 10))
        
        # Clear button (positioned to make it clear it clears both question and answer)
        clear_frame = ttk.Frame(tab_frame)
        clear_frame.grid(row=5, column=0, columnspan=3, pady=(5, 10))
        
        self.clear_qa_button = ttk.Button(clear_frame, text="Clear Question & Answer", command=self.clear_qa)
        self.clear_qa_button.pack()
        
        # Lock indicator (removed - no locking needed)
        
        # Bind tab change event (removed - no special handling needed)
        # self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
    
    def initialize_rag(self):
        """Initialize RAG system"""
        try:
            self.log("🚀 Initializing RAG System...")
            self.rag_system = RAGExcelProcessor()
            if self.rag_system.embeddings:
                self.log("✅ RAG system initialized successfully")
                self.status_var.set("Ready - RAG system loaded")
            else:
                self.log("❌ Failed to initialize RAG system")
                self.status_var.set("Error - RAG system failed to load")
        except Exception as e:
            self.log(f"❌ Error initializing RAG system: {e}")
            self.status_var.set("Error - Initialization failed")
    
    def ask_question(self):
        """Handle asking a question"""
        question = self.question_text.get("1.0", tk.END).strip()
        
        if not question:
            messagebox.showwarning("Warning", "Please enter a question")
            return
            
        if not self.rag_system or not self.rag_system.embeddings:
            messagebox.showerror("Error", "RAG system is not initialized")
            return
        
        # Disable ask button and show processing
        self.ask_button.config(state="disabled")
        self.qa_status_var.set("Processing question...")
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert("1.0", "🤔 Thinking...")
        
        # Process in separate thread to avoid blocking UI
        thread = threading.Thread(target=self.process_question, args=(question,))
        thread.daemon = True
        thread.start()
    
    def process_question(self, question):
        """Process the question in a separate thread"""
        try:
            # Get answer from RAG system
            answer = self.rag_system.get_rag_response(question)
            
            # Update UI in main thread
            self.root.after(0, self.display_answer, question, answer)
            
        except Exception as e:
            error_msg = f"❌ Error processing question: {e}"
            self.root.after(0, self.display_answer, question, error_msg)
    
    def display_answer(self, question, answer):
        """Display the answer in the UI"""
        # Clear and display new answer (only the answer, not the question)
        self.answer_text.delete("1.0", tk.END)
        
        # Format the response - only show the answer and timestamp
        formatted_response = f"{answer}\n\n"
        formatted_response += f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.answer_text.insert("1.0", formatted_response)
        
        # Store the answer for copying (without timestamp)
        self.current_answer = answer
        
        # Re-enable ask button and enable copy button
        self.ask_button.config(state="normal")
        self.copy_button.config(state="normal")
        self.qa_status_var.set("Ready for next question")
    
    def copy_answer(self):
        """Copy the current answer to clipboard"""
        if hasattr(self, 'current_answer') and self.current_answer:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.current_answer)
            self.qa_status_var.set("Answer copied to clipboard!")
            # Reset status message after 2 seconds
            self.root.after(2000, lambda: self.qa_status_var.set("Ready for next question"))
        else:
            messagebox.showwarning("Warning", "No answer to copy")
    
    def clear_qa(self):
        """Clear question and answer fields"""
        self.question_text.delete("1.0", tk.END)
        self.answer_text.delete("1.0", tk.END)
        self.copy_button.config(state="disabled")
        if hasattr(self, 'current_answer'):
            self.current_answer = ""
        self.qa_status_var.set("Ready for questions")
    
    def browse_file(self):
        """Browse for Excel file"""
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path_var.set(file_path)
    
    def log(self, message):
        """Add message to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_var.set(value)
        self.status_var.set(f"Processing... {value:.1f}%")
        self.root.update()
    
    def pause_processing(self):
        """Pause the current processing"""
        if self.rag_system:
            self.rag_system.pause_processing()
            self.pause_button.config(state="disabled")
            self.resume_button.config(state="normal")
            self.status_var.set("Paused")
            self.log("⏸️ Processing paused by user")
    
    def resume_processing(self):
        """Resume the paused processing"""
        if self.rag_system:
            self.rag_system.resume_processing()
            self.pause_button.config(state="normal")
            self.resume_button.config(state="disabled")
            self.log("▶️ Processing resumed by user")
    
    def stop_processing(self):
        """Stop the current processing"""
        if self.rag_system:
            self.rag_system.stop_processing()
            self.processing = False
            self.process_button.config(state="normal")
            self.pause_button.config(state="disabled")
            self.resume_button.config(state="disabled")
            self.stop_button.config(state="disabled")
            self.status_var.set("Stopped")
            self.log("⏹️ Processing stopped by user")
    
    def resume_from_temp(self):
        """Resume processing from a temporary file"""
        if not self.file_path_var.get():
            messagebox.showerror("Error", "Please select an Excel file first")
            return
        
        if self.processing:
            messagebox.showwarning("Warning", "Processing is already in progress")
            return
        
        # Find temporary files
        temp_files = self.rag_system._find_temp_files(self.file_path_var.get()) if self.rag_system else []
        
        if not temp_files:
            messagebox.showinfo("No Temp Files", "No temporary files found for this Excel file")
            return
        
        # Show selection dialog for temp files
        temp_options = [f"{tf['filename']} (up to row {tf['row_count']})" for tf in temp_files]
        
        # Create selection dialog
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Select Temporary File")
        selection_window.geometry("500x300")
        selection_window.transient(self.root)
        selection_window.grab_set()
        
        ttk.Label(selection_window, text="Select a temporary file to resume from:", 
                 font=("Arial", 12)).pack(pady=10)
        
        # Listbox for file selection
        listbox_frame = ttk.Frame(selection_window)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        listbox = tk.Listbox(listbox_frame)
        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        
        for option in temp_options:
            listbox.insert(tk.END, option)
        
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Select the latest file by default
        if temp_options:
            latest_index = temp_options.index(max(temp_options, key=lambda x: int(x.split('row ')[1].split(')')[0])))
            listbox.selection_set(latest_index)
        
        # Buttons
        button_frame = ttk.Frame(selection_window)
        button_frame.pack(pady=10)
        
        selected_temp_file = None
        
        def on_select():
            nonlocal selected_temp_file
            selection = listbox.curselection()
            if selection:
                selected_temp_file = temp_files[selection[0]]['path']
                selection_window.destroy()
        
        def on_cancel():
            selection_window.destroy()
        
        ttk.Button(button_frame, text="Resume", command=on_select).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        # Wait for selection
        self.root.wait_window(selection_window)
        
        if selected_temp_file:
            self.log(f"📂 Selected temporary file: {Path(selected_temp_file).name}")
            self.start_processing_with_temp(selected_temp_file)
    
    def start_processing_with_temp(self, temp_file_path):
        """Start processing with a specific temporary file"""
        if not self.rag_system or not self.rag_system.embeddings:
            messagebox.showerror("Error", "RAG system is not initialized")
            return
        
        # Start processing in separate thread
        self.processing = True
        self.process_button.config(state="disabled")
        self.pause_button.config(state="normal")
        self.stop_button.config(state="normal")
        self.resume_temp_button.config(state="disabled")
        
        thread = threading.Thread(target=self.process_file_from_temp, args=(temp_file_path,))
        thread.daemon = True
        thread.start()
    
    def process_file_from_temp(self, temp_file_path):
        """Process file starting from a temporary file"""
        try:
            file_path = self.file_path_var.get()
            continue_on_error = self.continue_on_error_var.get()
            auto_save_interval = int(self.auto_save_var.get())
            
            self.log(f"🔄 Resuming processing from temporary file")
            
            # Process the file starting from temp
            output_path = self.rag_system.process_excel_file(
                file_path,
                progress_callback=self.update_progress,
                log_callback=self.log,
                continue_on_error=continue_on_error,
                auto_save_interval=auto_save_interval,
                resume_from_temp=temp_file_path
            )
            
            if output_path:
                self.log(f"\n🎉 Processing completed successfully!")
                self.log(f"📄 Output file: {output_path}")
                self.status_var.set("Completed successfully")
                messagebox.showinfo("Success", f"Processing completed!\nOutput saved to:\n{output_path}")
            else:
                self.log("\n❌ Processing failed")
                self.status_var.set("Processing failed")
                messagebox.showerror("Error", "Processing failed. Check the log for details.")
                
        except Exception as e:
            self.log(f"\n❌ Unexpected error: {e}")
            self.status_var.set("Error occurred")
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
        
        finally:
            # Reset all control states
            self.processing = False
            self.process_button.config(state="normal")
            self.pause_button.config(state="disabled")
            self.resume_button.config(state="disabled")
            self.stop_button.config(state="disabled")
            self.resume_temp_button.config(state="normal")
            self.progress_var.set(0)
    
    def start_processing(self):
        """Start processing in a separate thread"""
        if not self.file_path_var.get():
            messagebox.showerror("Error", "Please select an Excel file first")
            return
        
        if not self.rag_system or not self.rag_system.embeddings:
            messagebox.showerror("Error", "RAG system is not initialized")
            return
        
        if self.processing:
            messagebox.showwarning("Warning", "Processing is already in progress")
            return
        
        # Start processing in separate thread
        self.processing = True
        self.process_button.config(state="disabled")
        self.pause_button.config(state="normal")
        self.stop_button.config(state="normal")
        self.resume_temp_button.config(state="disabled")
        self.browse_button.config(state="disabled")
        
        thread = threading.Thread(target=self.process_file)
        thread.daemon = True
        thread.start()
    
    def process_file(self):
        """Process the Excel file"""
        try:
            file_path = self.file_path_var.get()
            continue_on_error = self.continue_on_error_var.get()
            auto_save_interval = int(self.auto_save_var.get())
            
            self.log(f"🚀 Starting processing of: {Path(file_path).name}")
            self.log(f"⚙️ Continue on error: {continue_on_error}")
            self.log(f"💾 Auto-save interval: {auto_save_interval} rows")
            
            output_path = self.rag_system.process_excel_file(
                file_path,
                progress_callback=self.update_progress,
                log_callback=self.log,
                continue_on_error=continue_on_error,
                auto_save_interval=auto_save_interval
            )
            
            if output_path:
                self.log(f"\n🎉 Processing completed successfully!")
                self.log(f"📄 Output file: {output_path}")
                self.status_var.set("Completed successfully")
                messagebox.showinfo("Success", f"Processing completed!\nOutput saved to:\n{output_path}")
            else:
                self.log("\n❌ Processing failed")
                self.status_var.set("Processing failed")
                messagebox.showerror("Error", "Processing failed. Check the log for details.")
                
        except Exception as e:
            self.log(f"\n❌ Unexpected error: {e}")
            self.status_var.set("Error occurred")
            messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")
        
        finally:
            # Reset all control states
            self.processing = False
            self.process_button.config(state="normal")
            self.pause_button.config(state="disabled")
            self.resume_button.config(state="disabled")
            self.stop_button.config(state="disabled")
            self.resume_temp_button.config(state="normal")
            self.browse_button.config(state="normal")
            self.progress_var.set(0)

    def run(self):
        """Run the GUI"""
        self.root.mainloop()


def main():
    """Main function"""
    print("🚀 Starting RAG Excel Processor GUI")
    print("📋 Based on: rag_system v1 Semantic search - check sql injection.py")
    
    app = RAGExcelGUI()
    app.run()


if __name__ == "__main__":
    main()
