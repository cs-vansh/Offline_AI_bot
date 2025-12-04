# Offline AI Bot

A comprehensive RAG (Retrieval-Augmented Generation) system that processes Excel files, builds searchable indexes using txtai embeddings, and generates AI-powered responses using local Ollama LLM. This system works completely offline after initial setup.

## User Interface

#### Excel File Processing
<img width="600" height="472" alt="Screenshot 2025-12-04 at 19 08 06" src="https://github.com/user-attachments/assets/2623a84d-ae61-442d-80c7-1e42e42b7f71" />

#### Single Question Processing
<img width="600" height="472" alt="Screenshot 2025-12-04 at 19 08 14" src="https://github.com/user-attachments/assets/95355c26-068d-410f-a852-c529f59b6b8d" />


## 🌟 Features

- **Offline Operation**: Works completely offline using local AI models
- **Excel Processing**: Intelligent extraction and processing of Q&A data from Excel files
- **Semantic Search**: Fast semantic search using txtai embeddings
- **RAG System**: Combines retrieval with LLM generation for accurate answers
- **GUI Interface**: User-friendly interface for batch processing Excel files
- **Flexible Architecture**: Multiple scripts for different use cases

---

## 📁 Project Structure

### Core Files

#### 1. **build_index.py**
**Purpose**: Index builder for creating searchable txtai embeddings from Excel files

**Key Features**:
- Processes Excel files and extracts clean Q&A data
- Filters out junk data (serial numbers, IDs, auto-generated content)
- Automatically detects question and answer columns using flexible keyword matching
- Appends to existing indexes without overwriting
- Optimized for large datasets with minimal metadata bloat
- Supports multiple sheets and workbooks

**Use Case**: Run this to build or update your searchable index from Excel files

**Main Functions**:
- `extract_qa_from_excel()`: Extracts clean text entries from Excel files
- `build_index_from_files()`: Builds txtai index from multiple Excel files
- Smart column detection for questions/answers
- Junk data filtering patterns

---

#### 2. **rag_system v1 Semantic search - check sql injection.py**
**Purpose**: Core RAG system implementation with semantic search and SQL injection protection

**Key Features**:
- Integrates txtai embeddings with Ollama LLM
- Semantic search for retrieving relevant documents
- Context-aware answer generation
- SQL injection protection with conservative string cleaning
- Content mapping from Excel files
- Confidence scoring for retrieved results
- Interactive CLI interface for testing

**Use Case**: Standalone RAG system for question-answering with semantic search

**Main Components**:
- `RAGSystem` class: Complete RAG implementation
- Ollama integration for LLM responses
- Content retrieval and ranking
- SQL-safe query processing
- Interactive demo mode

---

#### 3. **quick_rag.py**
**Purpose**: Quick testing tool for RAG system functionality

**Key Features**:
- Fast testing of RAG pipeline
- Works with existing index
- Retrieves and displays relevant documents with scores
- Shows actual content from Excel files if available
- Generates Ollama-powered responses
- Color-coded result quality indicators (🟢🟡🟠)

**Use Case**: Quick validation and testing of your RAG system setup

**Functions**:
- `quick_rag_test()`: End-to-end RAG test
- Displays top-k relevant documents
- Shows retrieval scores and content previews
- Generates and displays LLM responses

---

#### 4. **sql_qa.py**
**Purpose**: SQL-based question answering using txtai's SQL capabilities

**Key Features**:
- SQL query interface for txtai embeddings
- Direct content retrieval using SQL queries
- Quality scoring for answers (Excellent/Good/Fair/Weak)
- Content browsing capabilities
- Interactive CLI for testing

**Use Case**: Alternative approach to question-answering using SQL queries on the index

**Main Features**:
- `SQLQuestionAnswerer` class
- SQL query generation for semantic search
- Content availability checking
- Multiple result retrieval
- Formatted output with quality indicators

---

#### 5. **pyinstall.py**
**Purpose**: PyInstaller configuration for building standalone executable

**Key Features**:
- Packages the GUI application into a standalone executable
- Windowed mode (no console window)
- Configured for macOS compatibility
- Creates "Offline AI bot" application

**Use Case**: Build a distributable executable version of the application

**Configuration**:
- Main script: `rag_excel_processor_gui.py`
- Windowed mode enabled
- Custom application name: "Offline AI bot"

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8+**
2. **Ollama** (for LLM responses)
   - Download from: https://ollama.ai
   - Pull a model: `ollama pull llama3.2:3b`
3. **Required Python packages**:
   ```bash
   pip install pandas txtai sentence-transformers pyinstaller tkinter openpyxl
   ```

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Ollama is installed and running
4. Pull an Ollama model (e.g., `ollama pull llama3.2:3b`)

### Basic Usage

#### 1. Build Index from Excel Files
```bash
python build_index.py
```
- Place your Excel files in a designated folder
- Script will extract Q&A data and build searchable index

#### 2. Test RAG System
```bash
python quick_rag.py
```
- Quick test of your RAG setup
- Validates index and Ollama integration

#### 3. SQL-based Querying
```bash
python sql_qa.py
```
- Interactive question-answering
- SQL query interface to index

---

## 🔧 Configuration

### Environment Variables
- `KMP_DUPLICATE_LIB_OK='TRUE'`: Fixes OpenMP conflicts
- `TOKENIZERS_PARALLELISM='false'`: Disables tokenizer parallelism warnings

### Default Settings
- **txtai model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Ollama model**: `llama3.2:3b` (configurable)
- **Index path**: `./index`
- **Temp files**: `~/Documents/TempFiles-OfflineAIBot`

---

## 📊 Workflow

1. **Prepare Excel Files**: Organize your Q&A data in Excel format
2. **Build Index**: Run `build_index.py` to create searchable embeddings
3. **Process with GUI**: Use `rag_excel_processor_gui.py` to add AI responses to Excel rows
4. **Query System**: Use `quick_rag.py` or `sql_qa.py` for interactive Q&A
5. **Deploy**: Build executable with `pyinstall.py` for distribution

---

## 🛠️ Technical Details

### txtai Integration
- Semantic search using sentence transformers
- Efficient vector similarity search
- Persistent index storage
- SQL query support

### Ollama Integration
- Local LLM inference
- Context-aware generation
- Streaming responses
- Model flexibility

### Data Processing
- Smart Excel parsing
- Junk data filtering
- Column auto-detection
- Multi-sheet support

---

## 📝 File Naming Conventions

- **Temp files**: `{original_name}_temp_{sheet}_{rows}rows_{timestamp}.xlsx`
- **Output files**: Auto-saved with processing progress

---

## 🐛 Troubleshooting

### Index Issues
- Delete `index` folder and rebuild using `build_index.py`

### Ollama Connection
- Ensure Ollama is running: `ollama list`
- Check model availability: `ollama pull llama3.2:3b`

### Excel Processing
- Ensure Excel files have proper Q&A structure
- Check for empty sheets or malformed data

---

## ⚠️ Important Notes

- **Offline Operation**: System works offline after initial model downloads
- **Performance**: Processing speed depends on Excel file size and Ollama model
- **Storage**: Index size scales with amount of processed data
- **Privacy**: All data processing happens locally, no cloud services used.
