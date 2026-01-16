============================================================
  RAG Document Analysis System
  Optimizing RAG for Querying Locally Stored Public 
  Procurement Documents using Small Language Models
============================================================

ABOUT THIS PROJECT
------------------
This is a Retrieval-Augmented Generation (RAG) system optimized 
for querying locally stored public procurement documents using 
Small Language Models (SLMs). The system runs entirely offline 
and is designed for portable deployment.

REPOSITORY SETUP
----------------
This repository contains the source code only. Large files 
(Python environment, Tesseract OCR, AI models) are not included 
due to size constraints. To run the application, you need to 
download the following components separately:

REQUIRED DOWNLOADS
------------------
1. Python Environment (python/)
   - Embedded Python 3.11 with all dependencies pre-installed
   - Download: [Contact maintainer for download link]
   - Extract to: ./python/

2. Tesseract OCR (tesseract/)
   - Required for scanned PDF support
   - Download: https://github.com/UB-Mannheim/tesseract/wiki
   - Extract to: ./tesseract/

3. AI Models (app/base_model/, app/embedding_model/, app/model/)
   - Base model: Qwen2.5-0.5B-Instruct
   - Embedding model: all-MiniLM-L6-v2
   - Fine-tuned adapter model
   - Download: [Contact maintainer for download link]
   - Extract to respective folders in ./app/

FOLDER STRUCTURE
----------------
RAG System.bat              - Main application launcher
Create Desktop Shortcut.bat - Creates a desktop shortcut
README.txt                  - This file
.gitignore                  - Git ignore rules

app/                        - Application source code
  app.py                    - Streamlit web interface
  desktop_app.py            - Desktop application wrapper
  rag_backend.py            - RAG processing backend
  config.json               - Configuration settings
  requirements.txt          - Python dependencies list
  
  base_model/               - [DOWNLOAD] Base language model
  embedding_model/          - [DOWNLOAD] Sentence embeddings
  model/                    - [DOWNLOAD] Fine-tuned adapter
  knowledge_base/           - [GENERATED] Vector database
  sessions/                 - [GENERATED] User sessions

python/                     - [DOWNLOAD] Embedded Python environment
tesseract/                  - [DOWNLOAD] OCR engine

QUICK START (After Downloads)
-----------------------------
1. Ensure all required components are downloaded and extracted
2. Double-click "RAG System.bat" to launch the application
3. (Optional) Run "Create Desktop Shortcut.bat" for desktop icon
4. (Optional) Run "tesseract\setup_tesseract.bat" for OCR support

SYSTEM REQUIREMENTS
-------------------
- Windows 10/11 (64-bit)
- 8 GB RAM minimum (16 GB recommended)
- 5 GB free disk space (after all downloads)
- NVIDIA GPU optional (for faster processing)

FIRST LAUNCH
------------
The first launch may take 2-3 minutes as the AI models are loaded
into memory. Subsequent launches will be faster.

DEVELOPMENT SETUP
-----------------
If you want to run from source without the portable package:

1. Install Python 3.11
2. Install dependencies: pip install -r app/requirements.txt
3. Install Tesseract OCR and add to PATH
4. Download models to respective folders
5. Run: python app/desktop_app.py

TROUBLESHOOTING
---------------
1. "Python not found" - Ensure python/ folder is properly extracted
2. "Module not found" - Check all dependencies are installed
3. "Model not found" - Verify model folders contain all files
4. Slow performance - Close other applications to free up RAM

LICENSE
-------
This project is for academic/research purposes.

============================================================
  Thesis: Optimizing RAG for Querying Locally Stored 
  Public Procurement Documents using Small Language Models
  
  Author: Dean Reight F. Billedo
  Date: January 2026
============================================================
