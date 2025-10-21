#!/usr/bin/env python3
"""
txtai Index Builder - Processes Excel files to create searchable indexes
Clean data extraction without metadata bloat - optimized for large datasets
Automatically appends to existing indexes without overwriting
"""

# Set environment variables BEFORE any imports to fix OpenMP conflicts
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import csv
import re
from pathlib import Path
from txtai.embeddings import Embeddings
from datetime import datetime
import time

# Define patterns for filtering junk data
JUNK_PATTERNS = [
    r'^\d+$',           # Pure numbers (1, 2, 3)
    r'^\d+\.$',         # Numbers with dot (1., 2., 3.)
    r'^0+\d+$',         # Zero-padded numbers (001, 002)
    r'^[A-Z]+\d+$',     # ID patterns (REQ001, DOC002)
    r'^\d+\.\d+$',      # Decimal numbers (1.0, 2.0)
    r'^\d{1,3}$',       # Short numbers (typical serial numbers)
]

JUNK_COLUMNS = [
    'sr', 'serial', 'no', 'number', 's.no', 'sno', 'index', 'id', 'uid',
    'sequence', 'count', 'row', 'entry', 'record', 'item', 'ref', 'reference'
]

# Define flexible matching for question/answer columns
QUESTION_KEYWORDS = [
    "question", "requirement", "context", "query", "domain", "details",
    "description", "expectation", "objective", "ask", "scenario",
    "goal", "purpose", "criteria", "observation", "statement", "check",
    "concern", "challenge", "topic", "require", "business need", "item"
]

ANSWER_KEYWORDS = [
    "answer", "response", "reply", "comment", "remarks", "justification",
    "explanation", "action taken", "control", "mitigation", "evidence",
    "status", "implementation", "solution", "plan", "procedure", "practice",
    "measure", "remediation", "description", "resolution", "compliance",
    "observation response", "what we do", "how", "status update", "supporting detail"
]

def is_junk_column(col_name):
    """Check if a column name indicates it contains junk data (serial numbers, IDs, etc.)"""
    col_lower = col_name.lower().strip()
    # Remove common punctuation and spaces
    col_clean = re.sub(r'[.\s_-]', '', col_lower)
    return any(junk in col_clean for junk in JUNK_COLUMNS)

def is_junk_value(value):
    """Check if a value looks like junk data (serial numbers, auto-generated IDs, etc.)"""
    value_str = str(value).strip()
    return any(re.match(pattern, value_str) for pattern in JUNK_PATTERNS)

def is_question_col(col_name):
    return any(keyword in col_name.lower() for keyword in QUESTION_KEYWORDS)

def is_answer_col(col_name):
    return any(keyword in col_name.lower() for keyword in ANSWER_KEYWORDS)

def extract_qa_from_excel(file_path):
    """Extract clean text entries from Excel files - data only, no metadata"""
    print(f"[INFO] Reading Excel: {file_path}")
    
    # Get workbook name for logging
    workbook_name = Path(file_path).name
    
    xl = pd.ExcelFile(file_path)
    all_entries = []
    total_skipped_sheets = 0

    for sheet_name in xl.sheet_names:
        print(f"\n[INFO] Processing sheet: {sheet_name}")
        try:
            df = xl.parse(sheet_name)
        except Exception as e:
            print(f"[WARN] Failed to read sheet '{sheet_name}': {e}")
            continue

        if df.empty or df.shape[1] == 0:
            print(f"[WARN] Sheet '{sheet_name}' is empty or has no columns. Skipping.")
            total_skipped_sheets += 1
            continue

        df = df.dropna(how='all')  # Remove empty rows

        extracted_count = 0
        filtered_count = 0
        skipped_rows = 0
        
        for index, row in df.iterrows():
            # Combine ONLY meaningful data values, filter out junk
            clean_values = []
            filtered_items = []
            
            for col in df.columns:
                cell_value = str(row[col]).strip()
                if not cell_value or cell_value.lower() == "nan":
                    continue
                
                # Track junk columns and values for reporting
                if is_junk_column(col):
                    filtered_items.append(f"{col}:{cell_value}")
                    continue
                    
                # Skip junk values (auto-generated numbers, IDs, etc.)
                if is_junk_value(cell_value):
                    filtered_items.append(f"{col}:{cell_value}")
                    continue
                
                # Add only meaningful data
                clean_values.append(cell_value)
            
            if clean_values:
                # Format as Question: Answer: based on first two columns
                if len(clean_values) >= 2:
                    # First column = Question, Second column = Answer, rest = additional context
                    question = clean_values[0]
                    answer = clean_values[1]
                    additional_context = clean_values[2:] if len(clean_values) > 2 else []
                    
                    # Create structured Q&A format
                    if additional_context:
                        clean_text = f"Question: {question} Answer: {answer} Context: {' '.join(additional_context)}"
                    else:
                        clean_text = f"Question: {question} Answer: {answer}"
                        
                elif len(clean_values) == 1:
                    # Single value - treat as general context
                    clean_text = f"Context: {clean_values[0]}"
                else:
                    # Fallback to original format
                    clean_text = " ".join(clean_values)
                
                all_entries.append(clean_text)
                extracted_count += 1
                
                # Count filtered junk
                if filtered_items:
                    filtered_count += len(filtered_items)
            else:
                # This row was completely filtered out
                skipped_rows += 1
                if skipped_rows <= 3:  # Show first few skipped rows for debugging
                    all_row_values = [str(row[col]).strip() for col in df.columns if str(row[col]).strip().lower() != "nan"]
                    print(f"[DEBUG] Row {index+1} completely filtered out: {all_row_values[:5]}...")
                    print(f"[DEBUG] Filtered items: {filtered_items[:3]}...")

        print(f"[INFO] Extracted {extracted_count} clean entries from sheet: {sheet_name}")
        if skipped_rows > 0:
            print(f"[INFO] Skipped {skipped_rows} rows that were completely filtered out as junk")
        if filtered_count > 0:
            print(f"[INFO] Filtered out {filtered_count} junk items (serial numbers, IDs, etc.)")

        print(f"[INFO] Total data entries extracted: {extracted_count} from sheet: {sheet_name}")
        print(f"[INFO] Total rows processed: {len(df)} | Extracted: {extracted_count} | Skipped: {skipped_rows}")
        
        # Get workbook name from file path
        workbook_name = Path(file_path).name
        print(f"✅ [COMPLETED] Sheet '{sheet_name}' from workbook '{workbook_name}' has been processed successfully!")
        print("-" * 80)

    print(f"\n[SUMMARY] Total entries extracted: {len(all_entries)}")
    if total_skipped_sheets > 0:
        print(f"[SUMMARY] Skipped {total_skipped_sheets} sheet(s) due to format issues.")
    return all_entries

def append_to_existing_index(entries, index_path="index"):
    """Append new entries to an existing index"""
    print(f"[INFO] Loading existing index from: {index_path}")
    
    try:
        # Load existing index with content support
        config = {
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "content": True,
            "objects": True
        }
        index = Embeddings(config)
        index.load(index_path)
        print(f"[INFO] Existing index loaded successfully")
        
        # Get the next available document ID using timestamp
        next_id = get_next_available_id(index_path)
        
        # Prepare new documents
        documents = []
        print(f"[INFO] Preparing {len(entries)} new documents for indexing...")
        for i, entry in enumerate(entries):
            documents.append((next_id + i, entry, None))
            # Efficient progress tracking for large datasets
            if len(entries) > 100 and (i + 1) % 1000 == 0:
                print(f"[PROGRESS] Prepared {i+1}/{len(entries)}: ID={next_id + i}")
            elif len(entries) <= 100 and ((i + 1) % 10 == 0 or i == len(entries) - 1):
                print(f"[PROGRESS] Prepared {i+1}/{len(entries)}: ID={next_id + i}")
        
        # Add new documents to existing index
        print(f"[INFO] Adding {len(documents)} new documents to existing index...")
        index.upsert(documents)  # Use upsert to add to existing index
        
        # Save updated index
        index.save(index_path)
        print(f"[INFO] Updated index saved to: {index_path}/")
        print(f"[SUCCESS] Successfully appended {len(entries)} new entries to existing index!")
        
        return index
        
    except Exception as e:
        print(f"[ERROR] Failed to append to existing index: {e}")
        print(f"[INFO] Creating new index instead...")
        return build_txtai_index(entries, index_path)

def build_txtai_index(entries, index_path="index"):
    """Build a new txtai index from scratch with content storage"""
    print(f"[INFO] Building new txtai index with {len(entries)} entries...")
    
    # Configure index to store content as well as embeddings
    config = {
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "content": True,  # This enables content storage
        "objects": True   # This enables object storage for retrieval
    }
    
    index = Embeddings(config)

    # Prepare all documents with efficient progress tracking
    documents = []
    print("[INFO] Preparing documents for indexing...")
    for i, entry in enumerate(entries):
        # Store both ID, text content, and metadata
        documents.append((i, entry, None))
        # Only show progress for large datasets to avoid spam
        if len(entries) > 100 and (i + 1) % 1000 == 0:
            print(f"[PROGRESS] Prepared {i+1}/{len(entries)} documents")
        elif len(entries) <= 100 and (i + 1) % 10 == 0:
            print(f"[PROGRESS] Prepared {i+1}/{len(entries)} documents")

    # Index all documents at once (this is the correct way)
    print(f"[INFO] Indexing {len(documents)} documents with content storage...")
    index.index(documents)  # Called once with all documents

    # Save the index
    index.save(index_path)
    print(f"[INFO] Index with content saved to: {index_path}/")
    
    return index

def smart_append_to_index(qa_pairs, index_path="index"):
    """Smart function that either creates new index or appends to existing one"""
    if not os.path.exists(index_path):
        print(f"[INFO] No existing index found at {index_path}. Creating new index...")
        return build_txtai_index(qa_pairs, index_path)
    
    print(f"[INFO] Existing index found at {index_path}. Appending new data...")
    return append_to_existing_index(qa_pairs, index_path)

def get_next_available_id(index_path="index"):
    """Get the next available document ID to avoid conflicts"""
    try:
        # Load existing index to check existing IDs
        index = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
        index.load(index_path)
        
        # For txtai, we'll use a timestamp-based approach to ensure uniqueness
        import time
        next_id = int(time.time() * 1000)  # Use millisecond timestamp
        print(f"[INFO] Starting new document IDs from: {next_id}")
        return next_id
        
    except Exception as e:
        print(f"[WARN] Could not determine next ID: {e}. Using timestamp-based ID.")
        import time
        return int(time.time() * 1000)

def test_index(index_path="index", test_queries=None):
    """Test the built index with sample queries"""
    print(f"\n[INFO] Testing index from: {index_path}")
    
    if not os.path.exists(index_path):
        print(f"[ERROR] Index not found at: {index_path}")
        return
    
    try:
        # Load the index
        index = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
        index.load(index_path)
        
        # Default test queries if none provided
        if test_queries is None:
            test_queries = [
                "What is the data policy?",
                "How do you manage access?",
                "Tell me about security measures"
            ]
        
        # Test each query
        for query in test_queries:
            print(f"\n[TEST] Query: '{query}'")
            results = index.search(query, 3)
            if results:
                for i, result in enumerate(results):
                    score = result[1] if len(result) > 1 else "N/A"
                    doc_id = result[0] if len(result) > 0 else "N/A"
                    print(f"  Result {i+1}: ID={doc_id}, Score={score:.3f}")
            else:
                print("  No results found")
        
        print(f"[SUCCESS] Index test completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Failed to test index: {e}")

def show_index_stats(index_path="index"):
    """Show statistics about the existing index"""
    if not os.path.exists(index_path):
        print(f"[INFO] No index found at {index_path}")
        return
    
    try:
        index = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
        index.load(index_path)
        
        # Try to get index statistics
        print(f"[INFO] Index statistics for '{index_path}':")
        print(f"  - Index directory: {os.path.abspath(index_path)}")
        
        # Check index files
        index_files = list(Path(index_path).glob("*"))
        print(f"  - Index files: {len(index_files)} files")
        
        for file in index_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"    - {file.name}: {size_mb:.2f} MB")
        
        print(f"[SUCCESS] Index loaded successfully!")
        
    except Exception as e:
        print(f"[ERROR] Failed to load index: {e}")

def export_entries_to_csv(entries, output_file="extracted_entries.csv"):
    """Export entries to CSV for verification - clean format only"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Text'])  # Simple header, no IDs or metadata
        
        for entry in entries:
            # Clean the entry to handle problematic characters
            cleaned_entry = str(entry).replace('\n', ' ').replace('\r', ' ').strip()
            # Replace multiple spaces with single space
            cleaned_entry = ' '.join(cleaned_entry.split())
            writer.writerow([cleaned_entry])
    
    print(f"[INFO] Exported {len(entries)} clean entries to {output_file}")
    return output_file

def main():
    print("=== txtai Index Builder (Auto-Append Mode) ===")
    print("This tool will automatically append to existing indexes without overwriting.")
    
    # Show current index status
    index_path = "index"
    print(f"\n{'='*60}")
    print("CURRENT INDEX STATUS")
    print('='*60)
    show_index_stats(index_path)
    
    print("\nEnter Excel file paths (one per line, empty line to finish):")
    
    input_paths = []
    while True:
        path = input("Excel file path: ").strip()
        if not path:
            break
        if not Path(path).exists():
            print(f"[WARN] File not found: {path}. Skipping.")
            continue
        input_paths.append(path)
    
    if not input_paths:
        print("[ERROR] No valid Excel files provided.")
        return

    # Extract entries from all files
    all_entries = []
    for file_path in input_paths:
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print('='*60)
        
        try:
            entries = extract_qa_from_excel(file_path)
            all_entries.extend(entries)
            print(f"[INFO] Added {len(entries)} entries from {Path(file_path).name}")
        except Exception as e:
            print(f"[ERROR] Failed to extract entries from {file_path}: {e}")
            continue

    if not all_entries:
        print("[WARNING] No entries extracted from any file. No index will be built.")
        return

    print(f"\n{'='*60}")
    print(f"SUMMARY: Total entries from all files: {len(all_entries)}")
    print('='*60)

    # Check if index exists and inform user
    if os.path.exists(index_path):
        print(f"[INFO] Existing index found at '{index_path}/'")
        print(f"[INFO] Will append {len(all_entries)} new entries to existing index")
    else:
        print(f"[INFO] No existing index found. Will create new index at '{index_path}/'")

    try:
        # Always use smart append (creates new or appends as needed)
        index = smart_append_to_index(all_entries, index_path)
        
        # Export to CSV with timestamp to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_output = f"output/entries_{timestamp}.csv"
        export_entries_to_csv(all_entries, csv_output)
        
        # Test the index to ensure it works
        print(f"\n{'='*60}")
        print("TESTING INDEX")
        print('='*60)
        test_index(index_path)
        
        print(f"\n✅ SUCCESS! Index updated with {len(all_entries)} entries")
        print(f"📁 Index location: {index_path}/")
        print(f"📄 CSV export: {csv_output}")
        
        # Show updated index stats
        print(f"\n{'='*60}")
        print("UPDATED INDEX STATUS")
        print('='*60)
        show_index_stats(index_path)
        
    except Exception as e:
        print(f"[ERROR] Failed to build index or export: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
