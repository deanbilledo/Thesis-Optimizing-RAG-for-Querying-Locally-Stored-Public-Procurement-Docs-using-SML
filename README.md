# RAG System - Setup Guide

**AI-Powered Document Analysis for Procurement Compliance**  
*by Paper ID 30*

---

## 📋 System Overview

This RAG (Retrieval-Augmented Generation) system provides intelligent document analysis with a focus on Philippine Government Procurement (RA 9184) compliance. It operates 100% offline with local AI models.

### Key Features
- ✅ Semantic search across all documents
- ✅ RA 9184 compliance checking
- ✅ Structured data extraction (ABC, PR Number, Delivery Period, etc.)
- ✅ Cross-document comparison
- ✅ Full audit trail
- ✅ Smart query caching
- ✅ Offline operation (no internet required after setup)
- ✅ Citation-based answers with sources

---

## 🗂️ Required Folder Structure

Before running the system, ensure your `C:\RAG-App\` folder contains these files and directories:

```
C:\RAG-App\
│
├── 📄 app.py                          # Main Streamlit UI application
├── 📄 rag_backend.py                  # RAG engine with session management
├── 📄 resources.py                    # Advanced features (extraction, audit, cache)
├── 📄 config.json                     # System configuration
├── 📄 main.py                         # CLI interface (optional)
├── 📄 README.md                       # This file
│
├── 📁 model/                          # LoRA adapter files (REQUIRED)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── added_tokens.json
│   ├── chat_template.jinja
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── tokenizer.model
│
├── 📁 base_model/                     # Base LLM model (REQUIRED)
│   ├── config.json
│   ├── generation_config.json
│   ├── model-*.safetensors (multiple files)
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── ... (other model files)
│
├── 📁 embedding_model/                # Sentence embedding model (REQUIRED)
│   ├── config.json
│   ├── modules.json
│   ├── pytorch_model.bin
│   ├── sentence_bert_config.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── ... (other embedding files)
│
├── 📁 knowledge_base/                 # RA 9184 compliance knowledge (OPTIONAL)
│   └── chroma_db/                     # Pre-populated knowledge base
│       ├── chroma.sqlite3
│       └── ... (vector database files)
│
├── 📁 sessions/                       # User sessions (AUTO-CREATED)
│   └── <session_id>/
│       ├── pdfs/                      # Uploaded documents
│       ├── chroma_db/                 # Document vector database
│       ├── audit/                     # Query logs
│       ├── cache/                     # Cached results
│       └── metadata.json
│
└── 📁 .venv/                          # Python virtual environment (AUTO-CREATED)
    └── ... (virtual environment files)
```

---

## ⚙️ Prerequisites

### 1. **Python 3.9 or higher**
Download from: https://www.python.org/downloads/

Verify installation:
```powershell
python --version
```

### 2. **Required Models** (Download Before Running)

#### Base LLM Model: `google/gemma-3-1b-it`
- **Size:** ~2.5 GB
- **Purpose:** Text generation and question answering
- **Location:** Must be in `base_model/` folder
- **Download:** https://huggingface.co/google/gemma-3-1b-it

#### Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
- **Size:** ~80 MB
- **Purpose:** Semantic search and similarity matching
- **Location:** Must be in `embedding_model/` folder
- **Download:** https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

#### LoRA Adapter
- **Size:** ~10 MB
- **Purpose:** Fine-tuned weights for procurement documents
- **Location:** Must be in `model/` folder
- **Source:** Provided with this system

---

## 🚀 Installation Steps

### Step 1: Clone or Extract the Project
```powershell
# If you received a ZIP file, extract it to C:\RAG-App\
# Or clone from repository
cd C:\
# Extract your files here
```

### Step 2: Verify Folder Structure
Check that you have all required folders:
```powershell
cd C:\RAG-App
dir
```

You should see:
- ✅ `app.py`
- ✅ `rag_backend.py`
- ✅ `resources.py`
- ✅ `config.json`
- ✅ `model/` folder (with 8+ files inside)
- ✅ `base_model/` folder (with model files)
- ✅ `embedding_model/` folder (with embedding files)

### Step 3: Create Virtual Environment
```powershell
cd C:\RAG-App
python -m venv .venv
```

### Step 4: Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` appear in your terminal prompt.

### Step 5: Install Dependencies
```powershell
pip install streamlit torch transformers sentence-transformers chromadb peft PyPDF2 numpy
```

**Expected packages:**
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
python -c "import streamlit, torch, transformers, sentence_transformers, chromadb, peft, PyPDF2; print('✅ All packages installed')"
```

### Step 7: Verify Models Are Present
```powershell
python -c "from pathlib import Path; base = Path('base_model').exists(); emb = Path('embedding_model').exists(); lora = Path('model').exists(); print(f'Base Model: {base}'); print(f'Embedding: {emb}'); print(f'LoRA: {lora}')"
```

All three should show `True`.

---

## ▶️ Running the Application

### Start the Web Interface
```powershell
cd C:\RAG-App
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The browser will open automatically at `http://localhost:8501`

### First-Time Launch
On first run, the system will:
1. Load the base model (~30 seconds)
2. Load the embedding model (~10 seconds)
3. Initialize the database (~5 seconds)

**Total first-time load: ~45 seconds**

Subsequent launches are faster (~10 seconds).

---

## 📚 Using the System

### 1. Create a Session
1. Click **"+ New Session"** in the sidebar
2. Enter a session name (e.g., "Project Analysis")
3. Click **"Create"**

### 2. Upload Documents
1. Go to **"Documents"** tab
2. Click **"Upload Documents"**
3. Select PDF files (max 6 files, 15 pages each, 20MB total)
4. Wait for processing (~10 seconds per document)

### 3. Ask Questions
1. Go to **"Chat"** tab
2. Type your question in natural language
3. Press Enter or click Send
4. View answer with sources and page numbers

### 4. Check Compliance (RA 9184)
1. Select a specific document from dropdown
2. Click **"🔍 Check Compliance"**
3. View extracted fields:
   - ABC (Approved Budget Cost)
   - PR Number
   - Delivery Period
   - Pre-Bid Conference
   - Bid Opening Date
   - Closing Date

### 5. Use Advanced Features
Go to **"Advanced"** tab:

**📊 Extract Data:**
- Automatically extract all procurement fields
- See page numbers where data was found

**🔄 Compare Documents:**
- Select 2 documents
- Compare ABC amounts, delivery periods
- Identify missing fields

**📋 Audit Trail:**
- View all queries with timestamps
- See document usage statistics
- Download audit report

**💾 Cache Stats:**
- Monitor cache performance
- View hit rate
- Clear cache if needed

---

## 🔧 Configuration

Edit `config.json` to customize:

```json
{
  "model": {
    "base_model": "google/gemma-3-1b-it",
    "temperature": 0.1,
    "max_new_tokens": 150
  },
  "embeddings": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
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

## 🐛 Troubleshooting

### Issue: "Module not found" error
**Solution:**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Issue: "Model files not found"
**Solution:**
Check that `base_model/`, `embedding_model/`, and `model/` folders exist and contain files.

### Issue: Out of memory error
**Solution:**
- Close other applications
- Reduce `max_pdfs_per_session` in `config.json`
- Use smaller documents

### Issue: Slow performance
**Solution:**
- First query is always slower (model loading)
- Enable query cache (already enabled by default)
- Use GPU if available (automatic detection)

### Issue: "ChromaDB error"
**Solution:**
```powershell
# Delete and recreate the database
Remove-Item -Recurse sessions/*/chroma_db
# Restart the app
```

---

## 📊 Performance Expectations

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
- First query: ~15 seconds
- Cached query: <0.01 seconds
- Document upload: ~10 seconds/document
- Compliance check: ~8 seconds
- Data extraction: ~5 seconds

**With GPU (NVIDIA):**
- First query: ~3-5 seconds
- Cached query: <0.01 seconds
- Document upload: ~5 seconds/document

---

## 📁 Sample Test Data

Test the system with provided sample documents in `test2/` folder:

```
test2/
├── 4 ITB Renovation of College of Home Economics Laboratory, Function Room.pdf
├── 5 ITB Procurement of Various Books for the University Library.pdf
├── 4 RFQ Procurement of Various Appliances for the College of Agriculture.pdf
├── 5 RFQ Procurement of LAPTOP for the College of Agriculture.pdf
├── 6 ITB Procurement of Laboratory Analysis of Various Foods Samples for the.pdf
├── 6 RFQ Procurement of Various Audio Visual and Office Equipment for the.pdf
└── test2-ground-truth.json  # Expected answers for verification
```

### Verification Test
1. Upload `4 ITB Renovation...` document
2. Advanced → Extract Data
3. Verify:
   - ABC: `Php 6,919,129.11` ✅
   - Delivery Period: `120 Calendar Days` ✅

---

## 🔒 Privacy & Security

### Data Storage
- All data stored locally in `sessions/` folder
- No cloud uploads
- No external API calls (after models are downloaded)

### Offline Operation
- System works 100% offline
- No internet required after setup
- Models run on your computer

### Data Deletion
To remove all session data:
```powershell
Remove-Item -Recurse sessions/*
```

---

## 📞 Support

### Documentation
- `README.md` - This file
- `IMPLEMENTATION.md` - Technical implementation details
- `ADVANCED_FEATURES.md` - Advanced feature documentation

### Common Questions

**Q: Can I use this without internet?**  
A: Yes, after models are downloaded, it works 100% offline.

**Q: What file formats are supported?**  
A: Currently PDF only. Documents should be text-based (not scanned images).

**Q: How many documents can I upload?**  
A: 6 documents per session, 15 pages each, 20MB total (configurable).

**Q: Is my data private?**  
A: Yes, everything runs locally. No data is sent to external servers.

**Q: Can I use a GPU?**  
A: Yes, if you have an NVIDIA GPU, it will be detected automatically.

---

## 📝 Version Information

**Version:** 1.0  
**Release Date:** December 2025  
**Author:** Paper ID 30  
**Python:** 3.9+  
**License:** MIT

---

## 🎉 Quick Start Checklist

Before your first run, verify:

- [ ] Python 3.9+ installed
- [ ] Virtual environment created (`.venv/`)
- [ ] Dependencies installed (`pip install ...`)
- [ ] `base_model/` folder exists with model files
- [ ] `embedding_model/` folder exists with embedding files
- [ ] `model/` folder exists with LoRA adapter
- [ ] `config.json` present
- [ ] Test documents available (optional)

**Ready? Start the app:**
```powershell
streamlit run app.py
```

Enjoy your AI-powered document analysis system! 🚀
