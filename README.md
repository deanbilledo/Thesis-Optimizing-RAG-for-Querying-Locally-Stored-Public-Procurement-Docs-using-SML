# RAG System for Public Procurement Document Analysis

**Optimizing RAG for Querying Locally Stored Public Procurement Documents using Small Language Models**

---

## System Overview

This Retrieval-Augmented Generation (RAG) system provides intelligent document analysis with a focus on Philippine Government Procurement (RA 9184) compliance. The system operates entirely offline using local AI models.

### Key Features

- Semantic search across all documents
- RA 9184 compliance checking
- Structured data extraction (ABC, PR Number, Delivery Period, etc.)
- Cross-document comparison
- Full audit trail
- Smart query caching
- Offline operation (no internet required after setup)
- Citation-based answers with sources

---

## Required Folder Structure

Before running the system, ensure your project folder contains these files and directories:

```
RAG-App/
|
|-- app.py                          # Main Streamlit UI application
|-- rag_backend.py                  # RAG engine with session management
|-- resources.py                    # Advanced features (extraction, audit, cache)
|-- config.json                     # System configuration
|-- main.py                         # CLI interface (optional)
|-- README.md                       # This file
|
|-- model/                          # LoRA adapter files (REQUIRED)
|   |-- adapter_config.json
|   |-- adapter_model.safetensors
|   |-- added_tokens.json
|   |-- chat_template.jinja
|   |-- special_tokens_map.json
|   |-- tokenizer_config.json
|   |-- tokenizer.json
|   |-- tokenizer.model
|
|-- base_model/                     # Base LLM model (REQUIRED)
|   |-- config.json
|   |-- generation_config.json
|   |-- model.safetensors
|   |-- tokenizer_config.json
|   |-- tokenizer.json
|   |-- (other model files)
|
|-- embedding_model/                # Sentence embedding model (REQUIRED)
|   |-- config.json
|   |-- modules.json
|   |-- model.safetensors
|   |-- sentence_bert_config.json
|   |-- tokenizer_config.json
|   |-- vocab.txt
|   |-- (other embedding files)
|
|-- knowledge_base/                 # RA 9184 compliance knowledge (OPTIONAL)
|   |-- chroma_db/                  # Pre-populated knowledge base
|       |-- chroma.sqlite3
|       |-- (vector database files)
|
|-- sessions/                       # User sessions (AUTO-CREATED)
|   |-- <session_id>/
|       |-- pdfs/                   # Uploaded documents
|       |-- chroma_db/              # Document vector database
|       |-- audit/                  # Query logs
|       |-- cache/                  # Cached results
|       |-- metadata.json
|
|-- .venv/                          # Python virtual environment (AUTO-CREATED)
    |-- (virtual environment files)
```

---

## Prerequisites

### 1. Python 3.9 or Higher

Download from: https://www.python.org/downloads/

Verify installation:
```powershell
python --version
```

### 2. Required Models

#### Base LLM Model: Gemma 3 1B IT
- **Size:** Approximately 2 GB
- **Purpose:** Text generation and question answering
- **Location:** Must be placed in `base_model/` folder

#### Embedding Model: all-MiniLM-L6-v2
- **Size:** Approximately 80 MB
- **Purpose:** Semantic search and similarity matching
- **Location:** Must be placed in `embedding_model/` folder

#### LoRA Adapter
- **Size:** Approximately 10 MB
- **Purpose:** Fine-tuned weights for procurement documents
- **Location:** Must be placed in `model/` folder

---

## Installation Steps

### Step 1: Clone or Extract the Project

```powershell
git clone https://github.com/deanbilledo/Thesis-Optimizing-RAG-for-Querying-Locally-Stored-Public-Procurement-Docs-using-SML.git
cd Thesis-Optimizing-RAG-for-Querying-Locally-Stored-Public-Procurement-Docs-using-SML
```

### Step 2: Verify Folder Structure

Check that you have all required folders:
```powershell
dir
```

You should see:
- `app.py`
- `rag_backend.py`
- `resources.py`
- `config.json`
- `model/` folder (with adapter files)
- `base_model/` folder (with model files)
- `embedding_model/` folder (with embedding files)

### Step 3: Create Virtual Environment

```powershell
python -m venv .venv
```

### Step 4: Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` appear in your terminal prompt.

### Step 5: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Core packages include:**
- `streamlit` - Web UI framework
- `torch` - PyTorch for deep learning
- `transformers` - Hugging Face transformers
- `sentence-transformers` - Embedding model
- `chromadb` - Vector database
- `peft` - Parameter-efficient fine-tuning (LoRA)
- `PyPDF2` - PDF processing
- `numpy` - Numerical operations

### Step 6: Verify Installation

```powershell
python -c "import streamlit, torch, transformers, sentence_transformers, chromadb, peft, PyPDF2; print('All packages installed successfully')"
```

### Step 7: Verify Models Are Present

```powershell
python -c "from pathlib import Path; base = Path('base_model').exists(); emb = Path('embedding_model').exists(); lora = Path('model').exists(); print(f'Base Model: {base}'); print(f'Embedding: {emb}'); print(f'LoRA: {lora}')"
```

All three should return `True`.

---

## Running the Application

### Start the Web Interface

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`

### First-Time Launch

On first run, the system will:
1. Load the base model (approximately 30 seconds)
2. Load the embedding model (approximately 10 seconds)
3. Initialize the database (approximately 5 seconds)

**Total first-time load:** Approximately 45 seconds

Subsequent launches are faster (approximately 10 seconds).

---

## Using the System

### 1. Create a Session

1. Click **"+ New Session"** in the sidebar
2. Enter a session name (e.g., "Project Analysis")
3. Click **"Create"**

### 2. Upload Documents

1. Navigate to the **"Documents"** tab
2. Click **"Upload Documents"**
3. Select PDF files (maximum 6 files, 15 pages each, 20MB total)
4. Wait for processing (approximately 10 seconds per document)

### 3. Ask Questions

1. Navigate to the **"Chat"** tab
2. Type your question in natural language
3. Press Enter or click Send
4. View answer with sources and page numbers

### 4. Check Compliance (RA 9184)

1. Select a specific document from the dropdown
2. Click **"Check Compliance"**
3. View extracted fields:
   - ABC (Approved Budget Cost)
   - PR Number
   - Delivery Period
   - Pre-Bid Conference
   - Bid Opening Date
   - Closing Date

### 5. Use Advanced Features

Navigate to the **"Advanced"** tab:

**Extract Data:**
- Automatically extract all procurement fields
- View page numbers where data was found

**Compare Documents:**
- Select 2 documents for comparison
- Compare ABC amounts, delivery periods
- Identify missing fields

**Audit Trail:**
- View all queries with timestamps
- See document usage statistics
- Download audit report

**Cache Statistics:**
- Monitor cache performance
- View hit rate
- Clear cache if needed

---

## Configuration

Edit `config.json` to customize system behavior:

```json
{
  "model": {
    "base_model": "base_model",
    "temperature": 0.1,
    "max_new_tokens": 150
  },
  "embeddings": {
    "model_name": "embedding_model",
    "dimension": 384
  },
  "documents": {
    "max_pdfs_per_session": 6,
    "max_pages_per_pdf": 15,
    "max_total_size_mb": 20
  },
  "sessions": {
    "max_sessions": 10,
    "auto_delete_oldest": true
  }
}
```

---

## Troubleshooting

### Issue: "Module not found" Error

**Solution:**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "Model files not found"

**Solution:**
Verify that `base_model/`, `embedding_model/`, and `model/` folders exist and contain the required files.

### Issue: Out of Memory Error

**Solution:**
- Close other applications
- Reduce `max_pdfs_per_session` in `config.json`
- Use smaller documents

### Issue: Slow Performance

**Solution:**
- First query is always slower due to model loading
- Query cache is enabled by default for faster repeated queries
- GPU acceleration is automatic if an NVIDIA GPU is available

### Issue: "ChromaDB error"

**Solution:**
```powershell
# Delete and recreate the database
Remove-Item -Recurse sessions/*/chroma_db
# Restart the application
```

---

## Performance Specifications

### Hardware Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 10 GB free
- OS: Windows 10/11

**Recommended:**
- CPU: 8+ cores
- RAM: 16 GB
- Storage: 20 GB free
- GPU: NVIDIA GPU with 4GB+ VRAM (optional, for faster inference)

### Response Times

**With CPU (Intel i5/i7):**
- First query: Approximately 15 seconds
- Cached query: Less than 0.01 seconds
- Document upload: Approximately 10 seconds per document
- Compliance check: Approximately 8 seconds
- Data extraction: Approximately 5 seconds

**With GPU (NVIDIA):**
- First query: Approximately 3-5 seconds
- Cached query: Less than 0.01 seconds
- Document upload: Approximately 5 seconds per document

---

## Privacy and Security

### Data Storage

- All data is stored locally in the `sessions/` folder
- No cloud uploads
- No external API calls (after initial model download)

### Offline Operation

- System operates 100% offline
- No internet connection required after setup
- All models run locally on your computer

### Data Deletion

To remove all session data:
```powershell
Remove-Item -Recurse sessions/*
```

---

## Frequently Asked Questions

**Q: Can I use this without internet?**  
A: Yes, after models are downloaded, the system works entirely offline.

**Q: What file formats are supported?**  
A: Currently PDF only. Documents should be text-based (not scanned images).

**Q: How many documents can I upload?**  
A: 6 documents per session, 15 pages each, 20MB total (configurable in config.json).

**Q: Is my data private?**  
A: Yes, all processing occurs locally. No data is transmitted to external servers.

**Q: Can I use a GPU?**  
A: Yes, NVIDIA GPUs are detected and utilized automatically if available.

---

## Version Information

- **Version:** 1.0
- **Python:** 3.9+
- **License:** MIT

---

## Quick Start Checklist

Before your first run, verify:

- [ ] Python 3.9+ installed
- [ ] Virtual environment created (`.venv/`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `base_model/` folder exists with model files
- [ ] `embedding_model/` folder exists with embedding files
- [ ] `model/` folder exists with LoRA adapter
- [ ] `config.json` present

**Ready to start:**
```powershell
streamlit run app.py
```
