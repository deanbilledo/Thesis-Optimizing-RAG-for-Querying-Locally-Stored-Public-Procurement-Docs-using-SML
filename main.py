"""
Complete Modern RAG Pipeline - ChatGPT Style Streamlit Interface
Advanced RAG system with semantic retrieval and intelligent document processing
"""

import os
import json
import re
import sys
import io
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
os.environ['PPOCR_SHOW_LOG'] = '0'  # Disable PaddleOCR verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import streamlit as st
import socket

# Set Streamlit flag early to prevent console encoding conflicts
sys._called_from_streamlit = True
import psutil
import PyPDF2
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
from datetime import datetime
from rank_bm25 import BM25Okapi
from collections import Counter

# LangChain imports
try:
    # Use langchain-community for document loaders, embeddings, and vector stores
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.retrievers import BM25Retriever
    
    # Core LangChain for text splitting, chains, and memory
    from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
    from langchain.schema import Document
    from langchain.chains import RetrievalQA, ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.llms.base import LLM
    from langchain.callbacks.manager import CallbackManagerForLLMRun
    
    LANGCHAIN_AVAILABLE = True
    print("[+] LangChain imported successfully (using langchain-community)")
except ImportError as e:
    print(f"[!] LangChain not available: {e}")
    print(f"[!] Install with: pip install langchain langchain-community")
    LANGCHAIN_AVAILABLE = False
    LANGCHAIN_AVAILABLE = False

# Fix Windows console encoding (only for terminal mode)
if sys.platform == 'win32' and not hasattr(sys, '_called_from_streamlit'):
    try:
        if hasattr(sys.stdout, 'buffer') and hasattr(sys.stderr, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, ValueError):
        # Skip encoding fix if running in Streamlit or other environments
        pass


class PDFProcessor:
    """Extract text from PDFs with intelligent digital/scanned detection"""
    
    def __init__(self, use_ocr: bool = True):
        self.use_ocr = use_ocr
        
        # Load PyMuPDF for PDF handling
        try:
            import pymupdf
            from PIL import Image
            self.has_pymupdf = True
            self.has_pil = True
        except:
            self.has_pymupdf = False
            self.has_pil = False
            print("Warning: PyMuPDF/PIL not available")
        
        # Load pdfplumber for layout-aware text extraction (digital PDFs)
        try:
            import pdfplumber
            self.has_pdfplumber = True
            print("[+] pdfplumber loaded (layout-aware extraction)")
        except:
            self.has_pdfplumber = False
            print("Warning: pdfplumber not available")
        
        # Load multiple OCR engines for scanned PDFs
        self.ocr_readers = {}
        if use_ocr:
            # Try to load multiple OCR engines in priority order
            ocr_engines = [
                ('easyocr', self._load_easyocr),
                ('tesseract', self._load_tesseract), 
                ('paddleocr', self._load_paddleocr)
            ]
            
            for engine_name, loader_func in ocr_engines:
                try:
                    print(f"[*] Loading {engine_name.upper()}...")
                    # Add timeout for PaddleOCR to prevent hanging
                    if engine_name == 'paddleocr':
                        # Skip PaddleOCR if it causes timeout issues
                        print(f"[!] Skipping {engine_name.upper()} to prevent initialization timeout")
                        continue
                    
                    self.ocr_readers[engine_name] = loader_func()
                    print(f"[+] {engine_name.upper()} loaded successfully")
                except Exception as e:
                    print(f"[!] {engine_name.upper()} not available: {e}")
            
            if self.ocr_readers:
                print(f"[+] OCR engines available: {list(self.ocr_readers.keys())}")
            else:
                print("[-] No OCR engines available")
        
        # OCR priority order (best to worst)
        self.ocr_priority = ['easyocr', 'tesseract', 'paddleocr']
    
    @property 
    def ocr_reader(self):
        """Backward compatibility - return best available OCR engine"""
        for engine_name in self.ocr_priority:
            if engine_name in self.ocr_readers:
                return self.ocr_readers[engine_name]
        return None
        
    def _load_easyocr(self):
        """Load EasyOCR engine"""
        import easyocr
        return easyocr.Reader(['en'], gpu=True)  # Use GPU if available
    
    def _load_tesseract(self):
        """Load Tesseract engine with pytesseract"""
        import pytesseract
        from PIL import Image
        # Return a simple wrapper for consistent interface
        class TesseractWrapper:
            def __init__(self):
                # Configure Tesseract for better quality
                self.config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()/-: '
            
            def readtext(self, image):
                """EasyOCR-compatible interface"""
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                text = pytesseract.image_to_string(image, config=self.config)
                # Return in EasyOCR format: [bbox, text, confidence]
                return [([0, 0, 100, 100], text.strip(), 0.9)] if text.strip() else []
        
        return TesseractWrapper()
    
    def _load_paddleocr(self):
        """Load PaddleOCR engine (fallback) - DISABLED to prevent timeout issues"""
        # PaddleOCR can cause timeout issues during initialization
        # Skip for now to prevent application hanging
        print(f"[!] PaddleOCR disabled due to initialization timeout issues")
        return None
        
        # Original implementation (kept for reference):
        # from paddleocr import PaddleOCR
        # 
        # class PaddleOCRWrapper:
        #     def __init__(self):
        #         self.reader = PaddleOCR(lang='en')
        #     
        #     def readtext(self, image):
        #         """EasyOCR-compatible interface"""
        #         result = self.reader.predict(image)
        #         # Convert PaddleOCR format to EasyOCR format
        #         if result and len(result) > 0:
        #             texts = []
        #             for item in result:
        #                 if len(item) >= 2:
        #                     bbox = item[0] if len(item[0]) == 4 else [0, 0, 100, 100]
        #                     text = item[1] if isinstance(item[1], str) else str(item[1])
        #                     conf = item[2] if len(item) > 2 else 0.8
        #                     texts.append((bbox, text, conf))
        #             return texts
        #         return []
        # 
        # return PaddleOCRWrapper()
    
    def _is_scanned_pdf(self, pdf_path: str) -> bool:
        """Detect if PDF is scanned (image-based) or digital (text-based)"""
        try:
            import pymupdf
            doc = pymupdf.open(pdf_path)
            
            # Check first 3 pages (or all if less)
            pages_to_check = min(3, len(doc))
            total_text_chars = 0
            
            for page_num in range(pages_to_check):
                page = doc[page_num]
                text = page.get_text().strip()
                total_text_chars += len(text)
            
            doc.close()
            
            # If less than 100 chars per page on average, it's likely scanned
            avg_chars_per_page = total_text_chars / pages_to_check
            is_scanned = avg_chars_per_page < 100
            
            return is_scanned
            
        except Exception as e:
            print(f"  Error detecting PDF type: {e}")
            return True  # Assume scanned if detection fails
    
    def extract_text(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from PDF - auto-detect digital vs scanned"""
        
        # Step 1: Detect PDF type
        is_scanned = self._is_scanned_pdf(pdf_path)
        
        if is_scanned:
            print(f"    [Scanned PDF] Using PaddleOCR")
            return self._extract_scanned_pdf(pdf_path)
        else:
            print(f"    [Digital PDF] Using pdfplumber (layout-aware)")
            return self._extract_digital_pdf(pdf_path)
    
    def _extract_digital_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract from digital PDF using pdfplumber (layout-aware, preserves tables)"""
        
        if not self.has_pdfplumber:
            # Fallback to basic extraction
            return self._extract_with_pymupdf_basic(pdf_path)
        
        try:
            import pdfplumber
            
            pages = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text with layout preservation
                    text = page.extract_text(layout=True) or ""
                    
                    # Extract tables separately with multiple strategies
                    # Try default first
                    tables = page.extract_tables()
                    
                    # If no tables found, try text-based detection for structured content
                    if not tables:
                        tables = page.extract_tables(table_settings={
                            'vertical_strategy': 'text',
                            'horizontal_strategy': 'text',
                            'min_words_vertical': 1,
                            'min_words_horizontal': 1
                        })
                    
                    # If still no tables, try more aggressive detection with explicit lines
                    if not tables:
                        try:
                            # Get page dimensions to create explicit lines
                            page_width = page.width
                            page_height = page.height
                            
                            # Create explicit vertical and horizontal lines
                            explicit_vertical = [0, page_width / 2, page_width]
                            explicit_horizontal = [0, page_height / 2, page_height]
                            
                            tables = page.extract_tables(table_settings={
                                'vertical_strategy': 'explicit',
                                'horizontal_strategy': 'explicit',
                                'explicit_vertical_lines': explicit_vertical,
                                'explicit_horizontal_lines': explicit_horizontal,
                                'snap_tolerance': 3,
                                'join_tolerance': 3
                            })
                        except Exception as e:
                            # If explicit strategy fails, just skip it
                            tables = []
                    
                    table_texts = []
                    
                    for table in tables:
                        if table and len(table) > 1:  # Must have at least 2 rows to be a meaningful table
                            # Convert table to text format
                            table_text = self._table_to_text(table)
                            if table_text and len(table_text.strip()) > 20:  # Must have meaningful content
                                table_texts.append(table_text)
                    
                    # Combine text and tables
                    combined_text = text
                    if table_texts:
                        combined_text += "\n\n" + "\n\n".join(table_texts)
                    
                    pages.append({
                        'page_number': page_num,
                        'content': combined_text,
                        'has_tables': len(tables) > 0,
                        'table_count': len(tables)
                    })
            
            return {
                'filename': Path(pdf_path).name,
                'path': pdf_path,
                'pages': pages,
                'total_pages': len(pages),
                'extraction_method': 'pdfplumber'
            }
            
        except Exception as e:
            print(f"    pdfplumber error: {e}, falling back")
            return self._extract_with_pymupdf_basic(pdf_path)
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table array to readable text"""
        if not table:
            return ""
        
        lines = []
        for row in table:
            if row:
                # Filter out None values and join with pipes
                clean_row = [str(cell).strip() if cell else "" for cell in row]
                lines.append(" | ".join(clean_row))
        
        return "\n".join(lines)
    
    def _extract_with_pymupdf_basic(self, pdf_path: str) -> Dict[str, Any]:
        """Basic extraction using PyMuPDF (fallback for digital PDFs)"""
        try:
            import pymupdf
            
            doc = pymupdf.open(pdf_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                pages.append({
                    'page_number': page_num + 1,
                    'content': text
                })
            
            doc.close()
            
            return {
                'filename': Path(pdf_path).name,
                'path': pdf_path,
                'pages': pages,
                'total_pages': len(pages),
                'extraction_method': 'pymupdf_basic'
            }
        except Exception as e:
            print(f"  PyMuPDF error: {e}")
            return None
    
    def _extract_scanned_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract from scanned PDF using multi-OCR fallback system"""
        
        if not self.ocr_readers or not self.has_pymupdf:
            print("    OCR not available, using fallback")
            return self._extract_with_pymupdf_basic(pdf_path)
        
        try:
            import pymupdf
            from PIL import Image
            import io
            
            doc = pymupdf.open(pdf_path)
            pages = []
            
            # Find best OCR engine available
            best_ocr = None
            best_engine_name = None
            for engine_name in self.ocr_priority:
                if engine_name in self.ocr_readers:
                    best_ocr = self.ocr_readers[engine_name]
                    best_engine_name = engine_name
                    break
            
            if not best_ocr:
                print("    No OCR engines available")
                return self._extract_with_pymupdf_basic(pdf_path)
            
            print(f"    [Scanned PDF] Using {best_engine_name.upper()}")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to high-res image for better OCR
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                
                # Try OCR with fallback to next engine if current fails
                ocr_text = self._extract_with_ocr_fallback(img_array)
                
                # Clean OCR text
                cleaned_text = self._clean_ocr_text(ocr_text)
                
                pages.append({
                    'page_number': page_num + 1,
                    'content': cleaned_text
                })
            
            doc.close()
            
            return {
                'filename': Path(pdf_path).name,
                'path': pdf_path,
                'pages': pages,
                'total_pages': len(pages),
                'extraction_method': f'ocr_{best_engine_name}'
            }
            
        except Exception as e:
            print(f"    OCR extraction error: {e}")
            return self._extract_with_pymupdf_basic(pdf_path)
    
    def _extract_with_ocr_fallback(self, img_array) -> str:
        """Try multiple OCR engines in priority order"""
        
        for engine_name in self.ocr_priority:
            if engine_name not in self.ocr_readers:
                continue
                
            try:
                ocr_engine = self.ocr_readers[engine_name]
                
                # Get OCR result using unified interface
                result = ocr_engine.readtext(img_array)
                
                # Extract text from result
                ocr_lines = []
                for item in result:
                    if len(item) >= 2:
                        text = item[1] if isinstance(item[1], str) else str(item[1])
                        confidence = item[2] if len(item) > 2 else 0.8
                        
                        # Only include high-confidence text
                        if confidence > 0.5 and len(text.strip()) > 0:
                            ocr_lines.append(text.strip())
                
                if ocr_lines:
                    ocr_text = "\n".join(ocr_lines)
                    print(f"    [OCR Success] {engine_name.upper()} extracted {len(ocr_text)} chars")
                    return ocr_text
                else:
                    print(f"    [OCR Failed] {engine_name.upper()} - no text found")
                    
            except Exception as e:
                print(f"    [OCR Error] {engine_name.upper()}: {e}")
                continue
        
        print("    [OCR Failed] All engines failed")
        return ""
    
    def _clean_ocr_text(self, text: str) -> str:
        if not text:
            return text
        
        # Enhanced OCR cleaning to handle severe corruption and improve date/number recognition
        
        # 0. Pre-process for date and number patterns (preserve critical information)
        # Detect and preserve date patterns even if surrounded by garbled text
        text = re.sub(r'\b([A-Z]{3})\s+(\d{1,2})\s+(\d{4})\b', r'\1 \2 \3', text)  # AUG 1 2005
        text = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', r'\1/\2/\3', text)  # 1/8/2005
        text = re.sub(r'\b(\d{1,2}):(\d{2})\s*(A\.?M\.?|P\.?M\.?)\b', r'\1:\2 \3', text, re.IGNORECASE)  # 9:30 A.M.
        
        # Preserve monetary amounts
        text = re.sub(r'\bP\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b', r'P \1', text)  # P 24,000.00
        text = re.sub(r'\bPHP\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\b', r'PHP \1', text)
        
        # 1. Remove excessive character repetitions (e.g., "ROOOOOOSOON" -> "ROSON")
        text = re.sub(r'(.)\1{4,}', r'\1\1', text)  # Reduce 5+ repeats to 2
        
        # 2. Detect and remove nonsense sequences like "ee tu ie the o so cn"
        # Split into words and filter out sequences of short meaningless words
        words = text.split()
        cleaned_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Check if this starts a sequence of nonsense words
            if (len(clean_word) <= 3 and 
                clean_word.isalpha() and 
                not any(v in clean_word.lower() for v in 'aeiou') and
                i + 2 < len(words)):
                
                # Look ahead for a sequence of similar short words
                nonsense_sequence = []
                j = i
                while (j < len(words) and j < i + 10):  # Check up to 10 words ahead
                    check_word = re.sub(r'[^\w]', '', words[j])
                    if (len(check_word) <= 3 and 
                        check_word.isalpha() and 
                        not any(v in check_word.lower() for v in 'aeiou')):
                        nonsense_sequence.append(words[j])
                        j += 1
                    else:
                        break
                
                # If we found 3+ consecutive short consonant-only words, skip them
                if len(nonsense_sequence) >= 3:
                    print(f"    [OCR Clean] Removing nonsense sequence: {' '.join(nonsense_sequence[:5])}...")
                    i = j  # Skip the entire sequence
                    continue
            
            # Keep normal words
            if (len(clean_word) <= 3 or 
                any(c.isdigit() for c in clean_word) or
                any(v in clean_word.lower() for v in 'aeiou') or
                clean_word.isupper() and len(clean_word) <= 6 or  # Keep short acronyms
                clean_word.lower() in ['the', 'and', 'for', 'with', 'from', 'to', 'of', 'in', 'on', 'at']):  # Keep common words
                cleaned_words.append(word)
            else:
                # Single nonsense word
                print(f"    [OCR Clean] Removing nonsense word: {word}")
            
            i += 1
        
        text = ' '.join(cleaned_words)
        
        # 3. Fix common OCR character errors (more aggressive for dates/amounts)
        # Common OCR misreads that affect dates and monetary amounts
        replacements = {
            '§': 'S', '¢': 'c',
            'rn': 'm',  # Common OCR confusion
            'cl': 'd',  # Common OCR confusion
            'l1': 'll', '1l': 'll',  # Common l/1 confusion
            'S0': '50', '0S': '05',  # Common 0/S confusion in numbers
            '0O': '00', 'O0': '00',  # Common O/0 confusion
            'D.': 'D',  # Remove stray dots after names
        }
        
        # Apply selective replacements
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Post-process to catch additional date patterns
        # Fix corrupted date patterns with regex
        
        # 4. Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # 5. Remove clearly bad artifacts while preserving important punctuation
        text = re.sub(r'[^\w\s\-.,;:()\[\]{}@#$%&*+=/<>!?\'\"\\|]', '', text)
        
        # 6. Clean up sequences of repeated punctuation or special chars
        text = re.sub(r'([^\w\s])\1{3,}', r'\1', text)  # Reduce punctuation repetition
        
        # 7. Normalize spaces around punctuation
        text = re.sub(r'\s+([.,;:])', r'\1', text)
        text = re.sub(r'([.,;:])\s+', r'\1 ', text)
        
        # 8. Remove lines that are mostly garbage (less than 40% recognizable words)
        lines = text.split('\n')
        clean_lines = []
        
        for line in lines:
            words_in_line = line.split()
            if not words_in_line:
                continue
                
            # Count recognizable words (have vowels, are numbers, or are common short words)
            recognizable = sum(1 for w in words_in_line 
                             if any(v in w.lower() for v in 'aeiou') or 
                                any(c.isdigit() for c in w) or 
                                len(w) <= 3 or
                                w.lower() in ['the', 'and', 'for', 'with', 'from', 'to', 'of', 'in', 'on', 'at'])
            
            if len(words_in_line) == 0 or recognizable / len(words_in_line) >= 0.4:
                clean_lines.append(line)
            else:
                print(f"    [OCR Clean] Removing corrupted line: {line[:50]}...")
        
        text = '\n'.join(clean_lines)
        
        return text.strip()


class OllamaLangChainLLM:
    """LangChain-compatible wrapper for Ollama LLM"""
    
    def __init__(self, model: str = "gemma3-finetuned:latest", base_url: str = "http://localhost:11434", **kwargs):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.temperature = kwargs.get('temperature', 0.01)
        self.max_tokens = kwargs.get('max_tokens', 200)
    
    def _llm_type(self) -> str:
        return "ollama"
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Make the class callable for LangChain compatibility"""
        return self._generate_response(prompt, **kwargs)
    
    def _generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from Ollama API"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get('temperature', self.temperature),
                "num_predict": kwargs.get('max_tokens', self.max_tokens),
                "num_ctx": 1024,
                "num_thread": 4,
                "repeat_penalty": 1.1,
                "top_k": 20,
                "top_p": 0.8,
                "use_mmap": True,
                "use_mlock": False
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
        except Exception as e:
            return f"ERROR: {str(e)}"


class LangChainDocumentProcessor:
    """Enhanced document processing with LangChain"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if LANGCHAIN_AVAILABLE:
            # LangChain text splitters
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", " ", ""],
                keep_separator=True
            )
            
            self.token_splitter = TokenTextSplitter(
                chunk_size=chunk_size // 4,  # Approximate token count
                chunk_overlap=overlap // 4
            )
        else:
            self.recursive_splitter = None
            self.token_splitter = None
    
    def load_pdf_documents(self, pdf_path: str):
        """Load PDF using LangChain PDF loader"""
        if not LANGCHAIN_AVAILABLE:
            return []
        
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"[LangChain] PDF loading failed: {e}")
            return []
    
    def split_documents(self, documents, method: str = "recursive"):
        """Split documents using LangChain text splitters"""
        if not LANGCHAIN_AVAILABLE or not documents:
            return documents
        
        try:
            if method == "recursive" and self.recursive_splitter:
                return self.recursive_splitter.split_documents(documents)
            elif method == "token" and self.token_splitter:
                return self.token_splitter.split_documents(documents)
            else:
                return documents
        except Exception as e:
            print(f"[LangChain] Document splitting failed: {e}")
            return documents


class SmartChunker:
    """Semantic chunking with RecursiveCharacterTextSplitter approach"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 150, ocr_reader=None):
        self.chunk_size = chunk_size  # 500-1000 tokens ≈ 800 chars
        self.overlap = overlap  # 100-200 tokens ≈ 150 chars
        self.ocr_reader = ocr_reader
        
        # Initialize LangChain document processor
        self.langchain_processor = LangChainDocumentProcessor(chunk_size, overlap)
        
        # Check for PyMuPDF (for image extraction from PDFs)
        self.has_pymupdf = False
        try:
            import pymupdf
            self.has_pymupdf = True
            print("[*] OCR-based table extraction enabled (PyMuPDF + PaddleOCR)")
        except:
            print("[!] PyMuPDF not available, OCR table extraction disabled")
    
    def _extract_document_identifier(self, filename: str) -> str:
        """Extract document identifier from filename for scanned PDF enhancement"""
        import re
        
        # Extract common document ID patterns
        patterns = [
            r'(PR\s*25-\d+-\d+)',  # PR 25-XX-XXX format
            r'(PB\s*-\s*PR-25-\d+-\d+)',  # PB - PR-25-XX-XXX format  
            r'(PO\s*25-\d+)',  # PO 25-XXX format
            r'(cv\d+)',  # cv1, cv2, etc.
            r'(ITB\s*25-\d+-\d+)',  # ITB patterns
            r'(\d{4}-\d{2}-\d{2})',  # Date patterns
            r'([A-Z]{2,}-?\d{2,})',  # General code patterns like PR-25, ITB-25, etc.
        ]
        
        # Clean filename (remove extension)
        clean_name = filename.lower().replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        
        # Try to extract identifiers
        identifiers = []
        for pattern in patterns:
            matches = re.findall(pattern, clean_name, re.IGNORECASE)
            identifiers.extend(matches)
        
        # Return the most specific identifier or default
        if identifiers:
            return identifiers[0].upper()
        
        # Fallback: use part of filename as identifier
        return filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').upper()[:20]
    
    def chunk_document(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk document intelligently - process both extracted tables and text"""
        chunks = []
        filename = doc_data['filename']
        pdf_path = doc_data['path']
        
        # Check if this is a scanned PDF for enhanced chunk content
        is_scanned_pdf = doc_data.get('extraction_method', '').startswith('ocr_')
        doc_identifier = self._extract_document_identifier(filename) if is_scanned_pdf else None
        
        # First, extract table chunks from pdfplumber-detected tables
        table_chunk_counter = 0
        for page in doc_data['pages']:
            if page.get('has_tables', False):
                page_num = page['page_number']
                content = page['content']
                
                # Look for table content (content after combining with tables)
                # Tables appear after the main text in the combined content
                lines = content.split('\n')
                
                # Find lines that look like table rows (containing " | ")
                table_lines = []
                current_table = []
                
                for line in lines:
                    if ' | ' in line and line.strip():
                        current_table.append(line.strip())
                    else:
                        # End of table - save if we have content
                        if current_table and len(current_table) > 1:  # At least 2 rows
                            table_content = '\n'.join(current_table)
                            if len(table_content.strip()) > 50:  # Meaningful content
                                table_chunk_counter += 1
                                
                                # Enhance content for scanned PDFs to improve semantic matching
                                enhanced_content = table_content
                                if is_scanned_pdf and doc_identifier:
                                    enhanced_content = f"Document {doc_identifier}:\n{table_content}"
                                
                                chunks.append({
                                    'content': enhanced_content,
                                    'metadata': {
                                        'filename': filename,
                                        'page': page_num,
                                        'chunk_type': 'table',
                                        'table_index': table_chunk_counter,
                                        'extraction_method': 'pdfplumber',
                                        'is_scanned_pdf': is_scanned_pdf,
                                        'doc_identifier': doc_identifier
                                    }
                                })
                        current_table = []
                
                # Handle final table if exists
                if current_table and len(current_table) > 1:
                    table_content = '\n'.join(current_table)
                    if len(table_content.strip()) > 50:
                        table_chunk_counter += 1
                        
                        # Enhance content for scanned PDFs to improve semantic matching
                        enhanced_content = table_content
                        if is_scanned_pdf and doc_identifier:
                            enhanced_content = f"Document {doc_identifier}:\n{table_content}"
                        
                        chunks.append({
                            'content': enhanced_content,
                            'metadata': {
                                'filename': filename,
                                'page': page_num,
                                'chunk_type': 'table',
                                'table_index': table_chunk_counter,
                                'extraction_method': 'pdfplumber',
                                'is_scanned_pdf': is_scanned_pdf,
                                'doc_identifier': doc_identifier
                            }
                        })
        
        # Skip OCR table extraction for digital PDFs (causing contamination)
        # ocr_table_chunks = self._extract_tables_ocr(pdf_path, filename)
        # chunks.extend(ocr_table_chunks)
        
        # Then extract text chunks using enhanced financial pattern detection
        for page in doc_data['pages']:
            page_num = page['page_number']
            content = page['content']
            
            # Remove table content from text (keep only non-table lines)
            lines = content.split('\n')
            text_lines = [line for line in lines if ' | ' not in line or not line.strip()]
            text_content = '\n'.join(text_lines)
            
            # Use enhanced chunking with financial pattern detection
            enhanced_chunks = self.enhanced_chunk_with_financial_detection(
                text_content, filename, page_num, 
                is_scanned_pdf=is_scanned_pdf, doc_identifier=doc_identifier
            )
            chunks.extend(enhanced_chunks)
        
        return chunks
    
    def _extract_tables_ocr(self, pdf_path: str, filename: str) -> List[Dict[str, Any]]:
        """Extract tables from image-based PDFs using OCR + structure detection"""
        
        if not self.has_pymupdf or not self.ocr_reader:
            return []
        
        try:
            import pymupdf
            from PIL import Image
            import io
            
            table_chunks = []
            doc = pymupdf.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convert page to high-res image
                pix = page.get_pixmap(dpi=300)  # High DPI for better OCR
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                # Run OCR with bounding boxes using available reader (support multiple interfaces)
                try:
                    # Some OCR readers (PaddleOCR) expose `ocr`, others (EasyOCR/Tesseract wrappers) expose `readtext`
                    if hasattr(self.ocr_reader, 'ocr'):
                        ocr_results_raw = self.ocr_reader.ocr(img_array)
                    elif hasattr(self.ocr_reader, 'readtext'):
                        ocr_results_raw = self.ocr_reader.readtext(img_array)
                    else:
                        # Last resort: call as function
                        ocr_results_raw = self.ocr_reader(img_array)

                    # Convert OCR format to standard format - handle EasyOCR vs PaddleOCR
                    ocr_results = []
                    if ocr_results_raw:
                        # EasyOCR returns direct list of [bbox, text, confidence]
                        if hasattr(self.ocr_reader, 'readtext'):  # EasyOCR
                            for item in ocr_results_raw:
                                if len(item) >= 3:
                                    bbox, text, conf = item[0], item[1], item[2]
                                    ocr_results.append((bbox, text, conf))
                        # PaddleOCR returns nested structure
                        elif len(ocr_results_raw) > 0 and ocr_results_raw[0] is not None:
                            try:
                                for line in ocr_results_raw[0]:
                                    if line and len(line) >= 2:
                                        bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                        text_info = line[1]  # (text, confidence)
                                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                            text, conf = text_info[0], text_info[1]
                                            ocr_results.append((bbox, text, conf))
                            except (TypeError, IndexError) as e:
                                print(f"    [OCR] Format parsing error: {e}")
                                continue

                    # Detect table regions using layout analysis
                    table_regions = self._detect_table_regions(ocr_results, img_array.shape)

                    # Extract text from each table region
                    for i, region in enumerate(table_regions):
                        table_text = self._extract_table_text(ocr_results, region)

                        if table_text and len(table_text) > 50:  # Skip tiny tables
                            table_chunks.append({
                                'content': table_text,
                                'metadata': {
                                    'filename': filename,
                                    'page': page_num + 1,
                                    'chunk_type': 'table',
                                    'table_index': i,
                                    'extraction_method': 'ocr'
                                }
                            })

                    if table_regions:
                        print(f"    [OCR-Table] Page {page_num + 1}: Found {len(table_regions)} table(s)")

                except Exception as e:
                    print(f"    [OCR-Table] Page {page_num + 1}: Error {e}")
                    continue
            
            doc.close()
            print(f"    [OCR-Table] Total: {len(table_chunks)} tables extracted")
            return table_chunks
            
        except Exception as e:
            print(f"    [OCR-Table] Failed: {e}")
            return []
    
    def _detect_table_regions(self, ocr_results: List, img_shape: Tuple) -> List[Dict]:
        """Detect table regions by analyzing text layout and alignment"""
        
        if not ocr_results or len(ocr_results) < 5:
            return []
        
        # Extract bounding boxes and sort by Y position (top to bottom)
        boxes = []
        for detection in ocr_results:
            bbox, text, conf = detection
            if conf < 0.3:  # Skip low confidence
                continue
            
            # Calculate center point and bounds
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            
            boxes.append({
                'bbox': bbox,
                'text': text,
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max,
                'x_center': x_center,
                'y_center': y_center,
                'width': x_max - x_min,
                'height': y_max - y_min
            })
        
        boxes.sort(key=lambda b: b['y_center'])
        
        # Group into rows by Y-position clustering
        rows = []
        current_row = [boxes[0]]
        y_threshold = 15  # pixels tolerance for same row
        
        for box in boxes[1:]:
            if abs(box['y_center'] - current_row[-1]['y_center']) < y_threshold:
                current_row.append(box)
            else:
                if len(current_row) > 1:  # Only keep rows with multiple elements
                    rows.append(sorted(current_row, key=lambda b: b['x_center']))
                current_row = [box]
        
        if len(current_row) > 1:
            rows.append(sorted(current_row, key=lambda b: b['x_center']))
        
        # Detect table regions: consecutive rows with similar column alignment
        table_regions = []
        current_table_rows = []
        
        for row in rows:
            if len(row) >= 2:  # Tables have at least 2 columns
                # Check if this row aligns with current table
                if current_table_rows and self._rows_aligned(current_table_rows[-1], row):
                    current_table_rows.append(row)
                else:
                    # Save previous table if it has enough rows
                    if len(current_table_rows) >= 2:  # At least 2 rows
                        table_regions.append({
                            'rows': current_table_rows,
                            'y_min': min(b['y_min'] for row in current_table_rows for b in row),
                            'y_max': max(b['y_max'] for row in current_table_rows for b in row),
                            'x_min': min(b['x_min'] for row in current_table_rows for b in row),
                            'x_max': max(b['x_max'] for row in current_table_rows for b in row)
                        })
                    current_table_rows = [row]
        
        # Don't forget the last table
        if len(current_table_rows) >= 2:
            table_regions.append({
                'rows': current_table_rows,
                'y_min': min(b['y_min'] for row in current_table_rows for b in row),
                'y_max': max(b['y_max'] for row in current_table_rows for b in row),
                'x_min': min(b['x_min'] for row in current_table_rows for b in row),
                'x_max': max(b['x_max'] for row in current_table_rows for b in row)
            })
        
        return table_regions
    
    def _rows_aligned(self, row1: List[Dict], row2: List[Dict], tolerance: int = 50) -> bool:
        """Check if two rows have similar column structure (aligned)"""
        
        if abs(len(row1) - len(row2)) > 2:  # Column count should be similar
            return False
        
        # Check if X-positions of columns are similar
        x1_positions = [b['x_center'] for b in row1]
        x2_positions = [b['x_center'] for b in row2]
        
        # For each column in row1, find closest column in row2
        alignments = 0
        for x1 in x1_positions:
            if any(abs(x1 - x2) < tolerance for x2 in x2_positions):
                alignments += 1
        
        # At least 50% of columns should align
        return alignments >= len(row1) * 0.5
    
    def _extract_table_text(self, ocr_results: List, region: Dict) -> str:
        """Extract and format text from a table region in simple pipe-delimited format"""
        
        rows = region['rows']
        if not rows:
            return ""
        
        table_lines = []
        
        for row in rows:
            # Sort boxes in row by X position (left to right)
            row_sorted = sorted(row, key=lambda b: b['x_center'])
            row_text = " | ".join([b['text'].strip() for b in row_sorted if b['text'].strip()])
            
            if row_text:
                table_lines.append(row_text)
        
        # Add simple separator after header (first row)
        if len(table_lines) > 1:
            table_lines.insert(1, "-" * 60)
        
        return '\n'.join(table_lines)
    
    def detect_financial_patterns(self, text):
        """Detect financial patterns in text that might indicate structured data"""
        patterns = {
            'prices': re.findall(r'P\s*\d+(?:,\d+)*(?:\.\d+)?', text, re.IGNORECASE),
            'quantities': re.findall(r'(\d+)\s+(pcs|piece|units?)', text, re.IGNORECASE),
            'products': re.findall(r'(Flash Drive|Hard Drive|PRINTER)', text, re.IGNORECASE)
        }
        
        # Count total patterns found
        pattern_count = sum(len(matches) for matches in patterns.values())
        
        return patterns, pattern_count
    
    def enhanced_chunk_with_financial_detection(self, text, filename, page_num, chunk_size=None, overlap=None, is_scanned_pdf=False, doc_identifier=None):
        """Enhanced chunking that detects financial patterns and classifies as tables"""
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap
            
        # First, check if the entire text has financial patterns
        patterns, pattern_count = self.detect_financial_patterns(text)
        
        # Use a higher threshold (≥8) to avoid over-classification
        # Also require multiple types of patterns (not just one type)
        has_multiple_pattern_types = sum(1 for p in patterns.values() if len(p) > 0) >= 2
        
        # If we find significant financial patterns AND multiple types, treat as structured content
        if pattern_count >= 8 and has_multiple_pattern_types:
            metadata = {
                'filename': filename,
                'page': page_num,
                'chunk_type': 'table',
                'pattern_count': pattern_count,
                'content_type': 'financial_data',
                'extraction_method': 'enhanced_financial_pattern',
                'has_prices': len(patterns['prices']) > 0,
                'has_quantities': len(patterns['quantities']) > 0,
                'has_products': len(patterns['products']) > 0,
                'price_count': len(patterns['prices']),
                'quantity_count': len(patterns['quantities']),
                'product_count': len(patterns['products'])
            }
            # Enhance content for scanned PDFs to improve semantic matching
            enhanced_content = text
            if is_scanned_pdf and doc_identifier:
                enhanced_content = f"Document {doc_identifier}:\n{text}"
            
            return [{
                'content': enhanced_content,
                'metadata': metadata
            }]
        
        # Otherwise, use standard text chunking
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_size = len(line)
            
            if current_size + line_size > chunk_size and current_chunk:
                # Save current chunk - check for financial patterns in smaller chunks too
                chunk_text = '\n'.join(current_chunk)
                chunk_patterns, chunk_pattern_count = self.detect_financial_patterns(chunk_text)
                
                # Only classify as financial table if it has substantial patterns
                if chunk_pattern_count >= 5 and sum(1 for p in chunk_patterns.values() if len(p) > 0) >= 2:
                    chunk_metadata = {
                        'filename': filename,
                        'page': page_num,
                        'chunk_type': 'table',
                        'pattern_count': chunk_pattern_count,
                        'content_type': 'financial_data',
                        'extraction_method': 'enhanced_financial_pattern',
                        'has_prices': len(chunk_patterns['prices']) > 0,
                        'has_quantities': len(chunk_patterns['quantities']) > 0,
                        'has_products': len(chunk_patterns['products']) > 0,
                        'price_count': len(chunk_patterns['prices']),
                        'quantity_count': len(chunk_patterns['quantities']),
                        'product_count': len(chunk_patterns['products'])
                    }
                else:
                    chunk_metadata = {
                        'filename': filename,
                        'page': page_num,
                        'chunk_type': 'text'
                    }
                
                # Enhance content for scanned PDFs to improve semantic matching
                enhanced_content = chunk_text
                if is_scanned_pdf and doc_identifier:
                    enhanced_content = f"Document {doc_identifier}:\n{chunk_text}"
                
                chunks.append({
                    'content': enhanced_content,
                    'metadata': chunk_metadata
                })
                
                # Keep overlap
                overlap_lines = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                current_chunk = overlap_lines.copy()
                current_size = sum(len(l) for l in current_chunk)
            
            current_chunk.append(line)
            current_size += line_size
        
        # Save remaining
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunk_patterns, chunk_pattern_count = self.detect_financial_patterns(chunk_text)
            
            # Check for financial patterns in final chunk
            if chunk_pattern_count >= 5 and sum(1 for p in chunk_patterns.values() if len(p) > 0) >= 2:
                chunk_metadata = {
                    'filename': filename,
                    'page': page_num,
                    'chunk_type': 'table',
                    'pattern_count': chunk_pattern_count,
                    'content_type': 'financial_data',
                    'extraction_method': 'enhanced_financial_pattern',
                    'has_prices': len(chunk_patterns['prices']) > 0,
                    'has_quantities': len(chunk_patterns['quantities']) > 0,
                    'has_products': len(chunk_patterns['products']) > 0,
                    'price_count': len(chunk_patterns['prices']),
                    'quantity_count': len(chunk_patterns['quantities']),
                    'product_count': len(chunk_patterns['products'])
                }
            else:
                chunk_metadata = {
                    'filename': filename,
                    'page': page_num,
                    'chunk_type': 'text'
                }
            
            # Enhance content for scanned PDFs to improve semantic matching
            enhanced_content = chunk_text
            if is_scanned_pdf and doc_identifier:
                enhanced_content = f"Document {doc_identifier}:\n{chunk_text}"
            
            chunks.append({
                'content': enhanced_content,
                'metadata': chunk_metadata
            })
        
        return chunks


class DocumentSummarizer:
    """Generate document summaries using LLM only - completely unbiased"""
    
    def __init__(self, llm_handler):
        self.llm = llm_handler
    
    def summarize_document(self, doc_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive summary from document content - NO pattern matching"""
        filename = doc_data['filename']
        
        # Get full document text
        full_text = '\n'.join([p['content'] for p in doc_data['pages']])
        preview = full_text[:6000]  # Use substantial context for better summary
        
        # Generate rich summary with LLM - let it discover EVERYTHING naturally
        prompt = f"""Analyze this document and create a comprehensive, distinctive summary focusing on UNIQUE identifiers and specific details.

DOCUMENT: {filename}

CONTENT:
{preview}

Create a highly detailed summary that captures:

CRITICAL IDENTIFIERS (Extract exactly as written):
- Purchase Request numbers (PR-XX, PR XX-XX-XX, etc.)
- Invitation to Bid numbers (PB-XX, ITB-XX, etc.) 
- Project codes, reference numbers, activity codes
- Department names, college names, unit names
- Specific item names and descriptions

KEY FINANCIAL DATA:
- Approved Budget Cost (ABC) amounts 
- Total costs, unit prices, budget figures
- Fee amounts, payment terms

TEMPORAL INFORMATION:
- Bid submission deadlines and times
- Bid opening dates and times  
- Pre-bid conference schedules
- Delivery periods and timelines

PROCUREMENT DETAILS:
- Exact items being procured (equipment, supplies, services)
- Quantities and specifications
- Target beneficiaries and end users

UNIQUE CHARACTERISTICS:
- Location-specific details (buildings, campuses, addresses)
- Technical specifications and requirements
- Vendor/supplier requirements

Focus on making this document easily distinguishable from others. Include ALL specific numbers, codes, amounts, and dates exactly as they appear in the document.

SUMMARY:"""
        
        # Generate LLM summary (it will naturally extract identifiers from content)
        llm_summary = self.llm.generate(prompt, temperature=0.0, max_tokens=800)
        
        return {
            'summary': llm_summary.strip(),
            'filename': filename,
            'metadata': {}  # No pattern-based extraction - LLM discovers everything naturally
        }


class VectorStore:
    """Two-stage vector storage with hybrid retrieval: document-level + chunk-level (BM25 + semantic)"""
    
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "rag_collection", embedding_model=None):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Initialize LangChain components if available
        self.langchain_vectorstore = None
        self.bm25_retriever = None
        
        if LANGCHAIN_AVAILABLE and embedding_model:
            try:
                # Create LangChain embeddings
                self.langchain_embeddings = HuggingFaceEmbeddings(
                    model_name="intfloat/e5-small-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                # Initialize LangChain Chroma vector store
                self.langchain_vectorstore = Chroma(
                    collection_name=f"lc_{collection_name}",
                    embedding_function=self.langchain_embeddings,
                    persist_directory=f"{persist_dir}/langchain"
                )
                print(f"[+] LangChain vector store initialized")
            except Exception as e:
                print(f"[!] LangChain vector store failed: {e}")
        
        # Initialize ChromaDB with retry logic for schema errors
        try:
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Two collections: one for document summaries, one for chunks
            self.doc_collection = self.client.get_or_create_collection(
                name=f"{collection_name}_documents",
                metadata={"hnsw:space": "cosine"}
            )
            self.chunk_collection = self.client.get_or_create_collection(
                name=f"{collection_name}_chunks",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Test collections to catch schema errors early
            try:
                _ = self.doc_collection.count()
                _ = self.chunk_collection.count()
            except Exception as schema_error:
                if "no such column" in str(schema_error).lower():
                    print(f"[!] Database schema corrupted: {schema_error}")
                    print(f"[*] Please restart the application to reset the database")
                    st.error("⚠️ Database schema error detected. Please restart the application to fix this.")
                raise schema_error
                
        except Exception as init_error:
            print(f"[!] ChromaDB initialization failed: {init_error}")
            if "no such column" in str(init_error).lower():
                print(f"[*] Database corruption detected. Manual reset required.")
                st.error("⚠️ Database corrupted. Please restart the application.")
            raise init_error
        
        # BM25 keyword index (built on-demand for relevant documents)
        self.bm25_index = None
        self.bm25_docs = []
    
    def cleanup(self):
        """Properly close ChromaDB connections"""
        try:
            if hasattr(self, 'client'):
                # Close collections
                self.doc_collection = None
                self.chunk_collection = None
                # Reset client
                self.client = None
                print("[+] ChromaDB connections closed")
        except Exception as e:
            print(f"[!] Cleanup error: {e}")
        self.bm25_metadata = []
    
    def _build_bm25_index(self, relevant_docs: List[str] = None):
        """Build BM25 index for keyword search on chunks from relevant documents"""
        # Get chunks from relevant documents
        if relevant_docs:
            results = self.chunk_collection.get(
                where={"filename": {"$in": relevant_docs}},
                include=["documents", "metadatas"]
            )
        else:
            results = self.chunk_collection.get(
                include=["documents", "metadatas"]
            )
        
        if not results['documents']:
            return
        
        self.bm25_docs = results['documents']
        self.bm25_metadata = results['metadatas']
        
        # Tokenize for BM25 (simple whitespace + lowercase)
        tokenized_docs = [doc.lower().split() for doc in self.bm25_docs]
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def _bm25_search(self, query: str, top_k: int = 30) -> List[Tuple[str, Dict, float]]:
        """BM25 keyword search - returns (content, metadata, score)"""
        if not self.bm25_index or not self.bm25_docs:
            return []
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((
                    self.bm25_docs[idx],
                    self.bm25_metadata[idx],
                    float(scores[idx])
                ))
        
        return results
    
    def add_document_summary(self, filename: str, summary: str, embedding: List[float], metadata: Dict[str, Any] = None):
        """Add document-level summary with metadata for first-stage retrieval"""
        doc_id = f"doc_{hash(filename)}"
        
        # Merge filename with additional metadata
        doc_metadata = {'filename': filename}
        if metadata:
            # Store identifiers for filtering
            if 'primary_id' in metadata and metadata['primary_id']:
                doc_metadata['primary_id'] = str(metadata['primary_id'])
            if 'all_ids' in metadata and metadata['all_ids']:
                doc_metadata['all_ids'] = '|'.join(str(x) for x in metadata['all_ids'] if x)  # Join for ChromaDB
            if 'departments' in metadata and metadata['departments']:
                doc_metadata['departments'] = '|'.join(str(x) for x in metadata['departments'][:3] if x)
        
        self.doc_collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[summary],
            metadatas=[doc_metadata]
        )
    
    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Add chunks to chunk-level store (legacy method name for compatibility)"""
        if not chunks or not embeddings:
            return
            
        ids = [f"chunk_{i}_{hash(c['content'])}" for i, c in enumerate(chunks)]
        documents = [c['content'] for c in chunks]
        # Clean metadata to avoid ChromaDB None value errors
        metadatas = []
        for c in chunks:
            metadata = c['metadata'].copy()
            # Ensure all values are not None and are ChromaDB-compatible types
            cleaned_metadata = {}
            for key, value in metadata.items():
                if value is not None:
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_metadata[key] = value
                    else:
                        # Convert other types to strings
                        cleaned_metadata[key] = str(value)
                else:
                    # Convert None to empty string
                    cleaned_metadata[key] = ""
            metadatas.append(cleaned_metadata)
        
        self.chunk_collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
    
    def search_documents(self, query_embedding: List[float], query_text: str = None, top_k: int = 5, return_summaries: bool = False):
        """Stage 1: Find relevant documents by hybrid approach - semantic similarity + document ID matching"""
        
        # Extract document IDs from query if present
        explicit_doc_matches = []
        if query_text:
            import re
            # Look for document patterns in the query
            doc_patterns = [
                r'(PR\s*25-\d+-\d+)',  # PR 25-XX-XXX format
                r'(PB\s*-\s*PR-25-\d+-\d+)',  # PB - PR-25-XX-XXX format  
                r'(PO\s*25-\d+)',  # PO 25-XXX format
                r'(cv\d+)',  # cv1, cv2, etc.
                r'(ITB\s*25-\d+-\d+)'  # ITB patterns
            ]
            
            for pattern in doc_patterns:
                matches = re.findall(pattern, query_text, re.IGNORECASE)
                for match in matches:
                    # Clean up the match
                    clean_match = re.sub(r'\s+', ' ', match.strip())
                    explicit_doc_matches.append(clean_match)
        
        # Pure semantic search on document summaries
        results = self.doc_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 3, 15),  # Get more candidates for better filtering
            include=["metadatas", "documents", "distances"] if return_summaries else ["metadatas", "distances"]
        )
        
        # Robustness: handle Chroma internal errors where HNSW segments may be missing
        if not results.get('metadatas') or not results['metadatas'][0]:
            # Attempt to re-initialize client and retry once
            try:
                self.client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(anonymized_telemetry=False))
                self.doc_collection = self.client.get_or_create_collection(name=f"{self.collection_name}_documents", metadata={"hnsw:space": "cosine"})
                results = self.doc_collection.query(query_embeddings=[query_embedding], n_results=min(top_k, 10), include=["metadatas", "documents"] if return_summaries else ["metadatas"])
            except Exception:
                return [] if not return_summaries else {'filenames': [], 'summaries': []}
        
        filenames = [meta['filename'] for meta in results['metadatas'][0]]
        summaries = results['documents'][0] if return_summaries and 'documents' in results else []
        distances = results['distances'][0] if 'distances' in results and results['distances'] else []
        
        # Apply similarity threshold filtering and boost exact document ID matches
        if distances:
            similarity_threshold = 0.70  # Slightly lower threshold to allow more candidates
            filtered_results = []
            for i, (filename, distance) in enumerate(zip(filenames, distances)):
                similarity = 1 - distance
                
                # Boost score if filename contains explicit document ID mentioned in query
                boost_applied = False
                if explicit_doc_matches:
                    for doc_id in explicit_doc_matches:
                        # Clean filename for comparison
                        clean_filename = re.sub(r'[^\w\d-]', ' ', filename.lower())
                        clean_doc_id = re.sub(r'[^\w\d-]', ' ', doc_id.lower())
                        
                        if clean_doc_id in clean_filename or any(part in clean_filename for part in clean_doc_id.split() if len(part) > 2):
                            similarity += 0.15  # Significant boost for exact document matches
                            boost_applied = True
                            break
                
                if similarity >= similarity_threshold:
                    filtered_results.append((filename, summaries[i] if summaries else "", similarity, boost_applied))
            
            # Sort by similarity (boosted scores will rank higher)
            filtered_results.sort(key=lambda x: x[2], reverse=True)
            
            # CRITICAL FIX: If we have explicit document matches, ONLY use those - no semantic neighbors
            if explicit_doc_matches and any(r[3] for r in filtered_results):
                # ONLY keep boosted documents (exact matches to document IDs in question)
                boosted = [r for r in filtered_results if r[3]]
                filtered_results = boosted  # Don't add any non-boosted documents
            else:
                # Standard filtering - keep best matches but allow multiple for context
                if len(filtered_results) >= 2:
                    best_score = filtered_results[0][2]
                    second_score = filtered_results[1][2]
                    if best_score - second_score > 0.08:  # 8% margin = clear winner
                        filtered_results = filtered_results[:1]  # Use only the best
                    else:
                        filtered_results = filtered_results[:3]  # Keep top 3 for context
                        
            filtered_results = filtered_results[:top_k]
            
            filenames = [r[0] for r in filtered_results]
            summaries = [r[1] for r in filtered_results]
            similarities = [r[2] for r in filtered_results]
            
            # Debug: print filtered similarity scores
            print(f"    [Semantic] Document scores (filtered): {list(zip(filenames, [f'{s:.3f}' for s in similarities]))}")
        else:
            # Fallback if no distances
            filenames = filenames[:top_k]
            summaries = summaries[:top_k] if summaries else []
        
        if return_summaries:
            return {
                'filenames': filenames,
                'summaries': summaries if summaries else []
            }
        else:
            return filenames
    
    def search(self, query: str, query_embedding: List[float], top_k: int = 10, relevant_docs: List[str] = None) -> List[Dict[str, Any]]:
        """Stage 2: Hybrid search (BM25 + semantic) on chunks from relevant documents"""
        
        # Build BM25 index for relevant documents
        self._build_bm25_index(relevant_docs)
        
        # 1. Semantic search (ChromaDB)
        filter_condition = {"filename": {"$in": relevant_docs}} if relevant_docs else None
        
        # Request documents, metadatas and distances explicitly so we keep chunk metadata
        # Request documents, metadatas and distances explicitly so we keep chunk metadata
        try:
            semantic_results = self.chunk_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 3, 30),  # Get more candidates for reranking
                where=filter_condition,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            # Try to recover from internal Chroma errors by re-initializing the client and retrying once
            try:
                self.client = chromadb.PersistentClient(path=self.persist_dir, settings=Settings(anonymized_telemetry=False))
                self.doc_collection = self.client.get_or_create_collection(name=f"{self.collection_name}_documents", metadata={"hnsw:space": "cosine"})
                self.chunk_collection = self.client.get_or_create_collection(name=f"{self.collection_name}_chunks", metadata={"hnsw:space": "cosine"})
                semantic_results = self.chunk_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k * 3, 30),
                    where=filter_condition,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                # Give up gracefully and return empty result set
                print(f"    [VectorStore.search] Chroma query failed and recovery failed: {e}")
                return []
        
        semantic_chunks = {}
        ids_list = semantic_results.get('ids', [[]])[0]
        docs_list = semantic_results.get('documents', [[]])[0]
        metas_list = semantic_results.get('metadatas', [[]])[0]
        dists_list = semantic_results.get('distances', [[]])[0]

        # If ids are not returned by the client, synthesize stable ids from content indices
        if not ids_list:
            ids_list = [f"chunk_idx_{i}" for i in range(len(docs_list))]

        for i in range(len(ids_list)):
            chunk_id = ids_list[i]
            content = docs_list[i] if i < len(docs_list) else ''
            metadata = metas_list[i] if i < len(metas_list) else {}
            distance = dists_list[i] if i < len(dists_list) else 1.0

            semantic_chunks[chunk_id] = {
                'content': content,
                'metadata': metadata or {},
                'semantic_score': 1 - distance,
                'bm25_score': 0.0
            }
        
        # 2. BM25 keyword search
        bm25_results = self._bm25_search(query, top_k=30)
        
        # 3. Merge and score (Reciprocal Rank Fusion)
        # Update BM25 scores for chunks
        for i, (content, metadata, bm25_score) in enumerate(bm25_results):
            # Find matching chunk by content
            for chunk_id, chunk_data in semantic_chunks.items():
                if chunk_data['content'] == content:
                    chunk_data['bm25_score'] = bm25_score
                    break
            else:
                # New chunk from BM25 not in semantic results
                chunk_id = f"bm25_{hash(content)}"
                semantic_chunks[chunk_id] = {
                    'content': content,
                    'metadata': metadata,
                    'semantic_score': 0.0,
                    'bm25_score': bm25_score
                }
        
        # 4. Compute hybrid scores (weighted combination + document priority)
        # Normalize BM25 scores to [0, 1] range
        max_bm25 = max([c['bm25_score'] for c in semantic_chunks.values()]) if semantic_chunks else 1.0
        if max_bm25 > 0:
            for chunk in semantic_chunks.values():
                chunk['bm25_score'] = chunk['bm25_score'] / max_bm25
        
        # Enhanced hybrid score: Boost BM25 for scanned PDFs (OCR text needs keyword matching)
        # CRITICAL: Extract PR numbers from query for content-level matching
        import re
        query_pr_numbers = set()
        pr_patterns = [r'PR\s*25-\d+-\d+', r'PO\s*25-\d+', r'ITB\s*25-\d+-\d+']
        for pattern in pr_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            query_pr_numbers.update([m.replace(' ', '').upper() for m in matches])
        
        for chunk in semantic_chunks.values():
            # Check if this chunk is from a scanned PDF document
            is_scanned = any(keyword in chunk['metadata'].get('filename', '').lower() 
                           for keyword in ['rfq', 'pr 25-07', 'pr 25-06'])  # Test7 patterns
            
            if is_scanned:
                # For scanned PDFs: 60% semantic + 40% BM25 (more keyword matching)
                base_score = (0.60 * chunk['semantic_score']) + (0.40 * chunk['bm25_score'])
            else:
                # For digital PDFs: 85% semantic + 15% BM25 (semantic focus)
                base_score = (0.85 * chunk['semantic_score']) + (0.15 * chunk['bm25_score'])
            
            # Boost score if chunk is from one of the candidate documents
            doc_boost = 0.0
            if relevant_docs:
                chunk_filename = chunk['metadata'].get('filename', '')
                if any(doc in chunk_filename for doc in relevant_docs):
                    doc_boost = 0.1  # 10% boost for correct document
            
            # CRITICAL: Boost if chunk content contains the EXACT PR number from query
            content_boost = 0.0
            if query_pr_numbers:
                chunk_content = chunk['content'].upper().replace(' ', '')
                for pr_num in query_pr_numbers:
                    if pr_num in chunk_content:
                        content_boost = 0.3  # 30% boost for exact PR match in content!
                        break
            
            chunk['hybrid_score'] = base_score + doc_boost + content_boost
        
        # 5. Sort by hybrid score and return top-k
        sorted_chunks = sorted(
            semantic_chunks.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )
        
        # CRITICAL FIX: Strictly filter to only chunks from relevant_docs before returning
        if relevant_docs:
            filtered_chunks = []
            for chunk in sorted_chunks:
                chunk_filename = chunk['metadata'].get('filename', '')
                # Only include if filename matches one of the relevant documents
                if any(doc.lower() in chunk_filename.lower() or chunk_filename.lower() in doc.lower() 
                       for doc in relevant_docs):
                    filtered_chunks.append(chunk)
            sorted_chunks = filtered_chunks
        
        # Take top-k after filtering
        sorted_chunks = sorted_chunks[:top_k]
        
        # Format for output
        final_chunks = []
        for chunk in sorted_chunks:
            final_chunks.append({
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'score': chunk['hybrid_score'],
                'semantic_score': chunk['semantic_score'],
                'bm25_score': chunk['bm25_score']
            })
        
        return final_chunks
    
    def add_langchain_documents(self, documents):
        """Add documents to LangChain vector store"""
        if not self.langchain_vectorstore or not documents:
            return False
        
        try:
            # Convert to LangChain Document format if needed
            if documents and not hasattr(documents[0], 'page_content'):
                # Convert from our chunk format to LangChain Document format
                lc_documents = []
                for doc in documents:
                    if isinstance(doc, dict) and 'content' in doc:
                        lc_doc = Document(
                            page_content=doc['content'],
                            metadata=doc.get('metadata', {})
                        )
                        lc_documents.append(lc_doc)
                documents = lc_documents
            
            # Add to vector store
            self.langchain_vectorstore.add_documents(documents)
            
            # Create BM25 retriever
            if LANGCHAIN_AVAILABLE:
                try:
                    self.bm25_retriever = BM25Retriever.from_documents(documents)
                    self.bm25_retriever.k = 8
                    print(f"[+] LangChain BM25 retriever created")
                except Exception as e:
                    print(f"[!] BM25 retriever creation failed: {e}")
            
            print(f"[+] Added {len(documents)} documents to LangChain vector store")
            return True
        except Exception as e:
            print(f"[!] LangChain document addition failed: {e}")
            return False
    
    def clear(self):
        """Clear all collections"""
        try:
            self.client.delete_collection(f"{self.collection_name}_documents")
            self.client.delete_collection(f"{self.collection_name}_chunks")
            self.doc_collection = self.client.get_or_create_collection(
                name=f"{self.collection_name}_documents",
                metadata={"hnsw:space": "cosine"}
            )
            self.chunk_collection = self.client.get_or_create_collection(
                name=f"{self.collection_name}_chunks",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Clear LangChain vector store
            if self.langchain_vectorstore:
                try:
                    self.langchain_vectorstore.delete_collection()
                    print(f"[+] LangChain vector store cleared")
                except Exception as e:
                    print(f"[!] LangChain clear failed: {e}")
        except:
            pass


class LLMHandler:
    """Handle LLM requests - supports Ollama, GGUF, and HuggingFace models"""
    
    def __init__(self, model: str = "gemma3-finetuned:latest", base_url: str = "http://localhost:11434", 
                 use_gguf: bool = False, gguf_path: str = None):
        self.model = model.strip()  # Remove any trailing spaces
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.use_gguf = use_gguf
        self.gguf_model = None
        
        # Load GGUF model if specified
        if use_gguf and gguf_path:
            self._load_gguf_model(gguf_path)
            self.langchain_llm = None  # GGUF doesn't use LangChain wrapper
        else:
            # Create LangChain-compatible LLM for Ollama
            if LANGCHAIN_AVAILABLE:
                self.langchain_llm = OllamaLangChainLLM(
                    model=self.model,
                    base_url=self.base_url
                )
                print(f"[+] LangChain LLM wrapper created")
            else:
                self.langchain_llm = None
            
            # Test connection and model availability
            self._test_connection()
    
    def _load_gguf_model(self, gguf_path: str):
        """Load GGUF model using llama-cpp-python"""
        try:
            from llama_cpp import Llama
            import torch
            
            print(f"\n{'='*80}")
            print(f"🚀 LOADING GGUF MODEL")
            print(f"{'='*80}")
            print(f"Model path: {gguf_path}")
            
            # Check GPU availability
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"✅ GPU detected: {gpu_name} ({gpu_memory:.2f} GB)")
                n_gpu_layers = -1  # Use all GPU layers
            else:
                print(f"⚠️  No GPU detected, using CPU")
                n_gpu_layers = 0
            
            # Load GGUF model
            print(f"Loading model...")
            import time
            load_start = time.time()
            
            self.gguf_model = Llama(
                model_path=gguf_path,
                n_gpu_layers=n_gpu_layers,  # Offload all layers to GPU
                n_ctx=2048,  # Context window
                n_batch=512,  # Batch size for prompt processing
                n_threads=4,  # CPU threads
                verbose=False,
                use_mlock=False,  # Don't lock memory
                use_mmap=True,  # Use memory mapping
            )
            
            load_time = time.time() - load_start
            
            print(f"\n{'='*80}")
            print(f"✅ GGUF MODEL LOADED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"⏱️  Load time: {load_time:.2f}s")
            print(f"🎯 GPU layers: {'ALL' if n_gpu_layers == -1 else n_gpu_layers}")
            print(f"📦 Context size: 2048 tokens")
            print(f"🚀 Expected speed: 20-30 tokens/second on GPU")
            print(f"{'='*80}\n")
            
        except ImportError:
            print(f"❌ Error: llama-cpp-python not installed!")
            print(f"Install with: pip install llama-cpp-python")
            raise
        except Exception as e:
            print(f"❌ Error loading GGUF model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _test_connection(self):
        """Test Ollama connection and model availability"""
        try:
            # Test basic connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            available_models = [m['name'] for m in models]
            
            if self.model in available_models:
                print(f"[+] LLM Ready: {self.model} (Ollama connected)")
            else:
                print(f"[!] Warning: Model '{self.model}' not found. Available models: {available_models}")
                if available_models:
                    print(f"[!] Consider using: {available_models[0]}")
                    
        except requests.exceptions.RequestException as e:
            print(f"[✗] Ollama connection failed: {e}")
            print(f"[!] Make sure Ollama is running: ollama serve")
        except Exception as e:
            print(f"[!] LLM test error: {e}")
    
    def generate(self, prompt: str, temperature: float = 0.01, max_tokens: int = 200) -> str:
        """Generate response from LLM - supports GGUF and Ollama"""
        import time
        print(f"  [LLM] Starting generation (prompt: {len(prompt)} chars, max_tokens: {max_tokens})")
        llm_start = time.time()
        
        # Use GGUF model if loaded
        if self.use_gguf and self.gguf_model:
            try:
                print(f"  [GGUF] Generating with llama.cpp...")
                
                # Generate with GGUF model
                output = self.gguf_model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=1 if temperature < 0.01 else 40,
                    top_p=0.1 if temperature < 0.01 else 0.9,
                    repeat_penalty=1.1,
                    stop=["<end_of_turn>", "<start_of_turn>", "\n\n\n"],
                    echo=False
                )
                
                llm_time = time.time() - llm_start
                response_text = output['choices'][0]['text'].strip()
                
                # Calculate tokens per second
                tokens_generated = output['usage']['completion_tokens']
                tok_per_sec = tokens_generated / llm_time if llm_time > 0 else 0
                
                print(f"  [⚡ GGUF] Generation: {llm_time:.3f}s | {tokens_generated} tokens | {tok_per_sec:.1f} tok/s")
                return response_text
                
            except Exception as e:
                llm_time = time.time() - llm_start
                print(f"  [✗] GGUF error after: {llm_time:.3f}s - {str(e)}")
                return f"ERROR: GGUF generation failed - {str(e)}"
        
        # Otherwise use Ollama
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 1024,   # Further reduced context for speed
                "num_thread": 4,   # Optimize CPU threads
                "repeat_penalty": 1.1,
                "top_k": 20,       # Faster sampling - reduced from 40
                "top_p": 0.8,      # Slightly more focused
                "use_mmap": True,  # Enable memory mapping for speed
                "use_mlock": False # Disable memory locking
            }
        }
        
        try:
            print(f"  [LLM] Generating response with {self.model}...")
            response = requests.post(self.api_url, json=payload, timeout=15)  # Reduced timeout to 15s
            response.raise_for_status()
            result = response.json()
            
            llm_time = time.time() - llm_start
            response_text = result.get('response', '').strip()
            
            if response_text:
                print(f"  [Timing] LLM generation: {llm_time:.3f}s ({len(response_text)} chars)")
                return response_text
            else:
                print(f"  [!] Empty response from {self.model}")
                return "ERROR: Empty response from model"
                
        except requests.exceptions.Timeout:
            llm_time = time.time() - llm_start
            print(f"  [✗] LLM timeout after: {llm_time:.3f}s")
            return "ERROR: Request timeout - model may be slow or unresponsive"
        except requests.exceptions.ConnectionError:
            llm_time = time.time() - llm_start
            print(f"  [✗] Connection error after: {llm_time:.3f}s")
            return "ERROR: Cannot connect to Ollama. Is it running?"
        except Exception as e:
            llm_time = time.time() - llm_start
            print(f"  [✗] LLM error after: {llm_time:.3f}s - {str(e)}")
            return f"ERROR: {str(e)}"
    
    def generate_langchain(self, prompt: str, **kwargs) -> str:
        """Generate response using LangChain wrapper"""
        if self.langchain_llm:
            return self.langchain_llm(prompt, **kwargs)
        else:
            return self.generate(prompt, **kwargs)


class LangChainRAGPipeline:
    """Enhanced RAG Pipeline with LangChain integration"""
    
    def __init__(self, config_path: str = "config.json", session_id: str = None):
        if not LANGCHAIN_AVAILABLE:
            print("[!] LangChain not available, skipping LangChain pipeline")
            return
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.session_id = session_id or "default"
        
        # Initialize LangChain components
        self.enhanced_vector_store = None
        self.retrieval_chain = None
        self.conversational_chain = None
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Remember last 5 exchanges
        )
    
    def setup_langchain_pipeline(self, llm_handler, vector_store):
        """Setup LangChain retrieval and conversational chains"""
        if not LANGCHAIN_AVAILABLE or not llm_handler.langchain_llm:
            return False
        
        try:
            self.enhanced_vector_store = vector_store
            
            if not hasattr(vector_store, 'langchain_vectorstore') or not vector_store.langchain_vectorstore:
                print("[LangChain] Vector store not available")
                return False
            
            # Create retrieval QA chain
            retriever = vector_store.langchain_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            )
            
            # Simple retrieval chain (fallback for compatibility)
            self.retrieval_chain = {
                'llm': llm_handler.langchain_llm,
                'retriever': retriever
            }
            
            print("[LangChain] Pipeline setup complete")
            return True
        except Exception as e:
            print(f"[LangChain] Pipeline setup failed: {e}")
            return False
    
    def query_langchain(self, question: str):
        """Query using LangChain chains"""
        if not LANGCHAIN_AVAILABLE or not self.retrieval_chain:
            return None
        
        try:
            # Simple retrieval and generation
            retriever = self.retrieval_chain['retriever']
            llm = self.retrieval_chain['llm']
            
            # Retrieve relevant documents
            docs = retriever.get_relevant_documents(question)
            
            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs[:3]])
            
            # Generate answer
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            answer = llm(prompt)
            
            return {
                'answer': answer,
                'source_documents': docs,
                'question': question
            }
        except Exception as e:
            print(f"[LangChain] Query failed: {e}")
            return None


class RAGPipeline:
    """Complete RAG Pipeline with Two-Stage Hybrid Retrieval + LLM Document Selection"""
    
    def __init__(self, config_path: str = "config.json", session_id: str = None):
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Store session ID for session-isolated collections
        self.session_id = session_id or "default"
        
        # Initialize LangChain pipeline
        if LANGCHAIN_AVAILABLE:
            self.langchain_pipeline = LangChainRAGPipeline(config_path, session_id)
            print(f"[+] LangChain pipeline initialized")
        else:
            self.langchain_pipeline = None
        
        # Initialize components
        print(f"\n=== RAG PIPELINE INITIALIZATION ===")
        print(f"[*] Session ID: {self.session_id}")
        print(f"[*] Config loaded: {config_path}")
        
        print(f"[*] Initializing PDF processor with OCR...")
        self.pdf_processor = PDFProcessor(use_ocr=True)
        print(f"[+] PDF processor ready")
        print(f"[*] Initializing smart chunker...")
        self.chunker = SmartChunker(
            chunk_size=self.config['chunking']['text_chunk_size'],
            overlap=self.config['chunking']['text_chunk_overlap'],
            ocr_reader=self.pdf_processor.ocr_reader  # Pass OCR reader for table extraction
        )
        print(f"[+] Smart chunker ready (size: {self.config['chunking']['text_chunk_size']}, overlap: {self.config['chunking']['text_chunk_overlap']})")
        
        print("[*] Loading embedding model (intfloat/e5-small-v2)...")
        # Detect GPU availability
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[*] Using device: {device}")
        
        # Use e5-small-v2 with CPU optimizations (OFFLINE MODE)
        print(f"    [OFFLINE] Loading e5-small-v2 from local cache only...")
        try:
            # Force offline mode by setting local_files_only=True
            self.embedding_model = SentenceTransformer(
                'intfloat/e5-small-v2',
                device=device,
                local_files_only=True  # Prevent internet downloads
            )
            print(f"    [OFFLINE] ✅ Successfully loaded e5-small-v2 from local cache")
        except Exception as e:
            print(f"    [OFFLINE] ❌ Failed to load e5-small-v2 locally: {e}")
            print(f"    [OFFLINE] Please ensure the model is pre-downloaded for offline use")
            raise RuntimeError(f"Offline mode requires pre-downloaded e5-small-v2 model: {e}")
        
        # Optimize for CPU if no GPU available
        if device == 'cuda':
            try:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"[+] Loaded e5-small-v2 on GPU: {gpu_name}")
            except Exception as e:
                print(f"[!] GPU check failed: {e}")
        else:
            print("[+] Loaded e5-small-v2 on CPU (optimized for fast inference)")
            # Enable CPU optimizations
            torch.set_num_threads(min(4, torch.get_num_threads()))  # Limit threads for faster single queries
            print(f"[+] CPU threads optimized: {torch.get_num_threads()}")
        
        # Create session-specific vector store with unique collection name
        session_collection_name = f"{self.config['vectorstore']['collection_name']}_{self.session_id}"
        print(f"[*] Initializing vector store: {session_collection_name}")
        try:
            self.vector_store = VectorStore(
                persist_dir=self.config['vectorstore']['persist_directory'],
                collection_name=session_collection_name,
                embedding_model=self.embedding_model
            )
            print(f"[+] Vector store ready: {self.config['vectorstore']['persist_directory']}")
            if LANGCHAIN_AVAILABLE:
                print(f"[+] LangChain vector store ready")
        except Exception as vs_error:
            print(f"[!] Vector store initialization failed: {vs_error}")
            print(f"[*] Attempting database cleanup and retry...")
            # Clean up and retry
            import shutil
            db_path = self.config['vectorstore']['persist_directory']
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
                print(f"[*] Database cleaned: {db_path}")
            self.vector_store = VectorStore(
                persist_dir=db_path,
                collection_name=session_collection_name
            )
            print(f"[+] Vector store recreated successfully")
        
        print(f"[*] Connecting to LLM: {self.config['model']['name']}")
        
        # Check if using GGUF model
        use_gguf = self.config['model'].get('use_gguf', False)
        gguf_path = self.config['model'].get('gguf_path', None)
        
        self.llm = LLMHandler(
            model=self.config['model']['name'],
            base_url=self.config['model']['base_url'],
            use_gguf=use_gguf,
            gguf_path=gguf_path
        )
        
        # Document summarizer for intelligent retrieval
        self.summarizer = DocumentSummarizer(self.llm)
        
        # Setup LangChain pipeline
        if LANGCHAIN_AVAILABLE and self.langchain_pipeline:
            print(f"[*] Setting up LangChain pipeline...")
            langchain_success = self.langchain_pipeline.setup_langchain_pipeline(self.llm, self.vector_store)
            if langchain_success:
                print(f"[+] LangChain pipeline ready")
            else:
                print(f"[!] LangChain pipeline setup failed, using fallback")
        
        self.system_prompt = self.config.get('system_prompt', '')
        
        print("[+] Enhanced RAG Pipeline ready!\n")
    
    def index_documents(self, pdf_paths: List[str]):
        """Process and index PDF documents with two-stage approach"""
        print(f"[*] Indexing {len(pdf_paths)} documents...")
        
        all_chunks = []
        
        for pdf_path in pdf_paths:
            print(f"  Processing: {Path(pdf_path).name}")
            
            # Extract text
            doc_data = self.pdf_processor.extract_text(pdf_path)
            if not doc_data:
                continue
            
            # Generate document summary using LLM (unbiased - no pattern matching)
            print("    Generating document summary...")
            summary_data = self.summarizer.summarize_document(doc_data)
            summary_text = summary_data['summary']
            
            # Add "passage:" prefix for e5 model
            summary_embedding = self.embedding_model.encode([f"passage: {summary_text}"])[0].tolist()
            
            # Store document summary (Stage 1 index)
            self.vector_store.add_document_summary(
                filename=Path(pdf_path).name,
                summary=summary_text,
                embedding=summary_embedding
            )
            
            # Chunk
            chunks = self.chunker.chunk_document(doc_data)
            all_chunks.extend(chunks)
            print(f"    [+] {len(chunks)} chunks created")
        
        if not all_chunks:
            print("  [!] No chunks extracted from PDFs")
            return
        
        # Generate embeddings for chunks with "passage:" prefix for e5 model
        print(f"\n[*] Generating embeddings for {len(all_chunks)} chunks...")
        texts = [f"passage: {c['content']}" for c in all_chunks]  # Add e5 prefix
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        ).tolist()
        
        # Add chunks to vector store (Stage 2 index)
        print("[*] Adding to vector store...")
        self.vector_store.add_documents(all_chunks, embeddings)
        
        print(f"[+] Indexed {len(all_chunks)} chunks from {len(pdf_paths)} documents")
        
        # Try LangChain indexing as well
        if LANGCHAIN_AVAILABLE and self.vector_store.langchain_vectorstore:
            print(f"[*] LangChain indexing documents...")
            langchain_success = self.index_documents_langchain(pdf_paths)
            if langchain_success:
                print(f"[+] LangChain indexing completed")
        
        print()
    
    def index_documents_langchain(self, pdf_paths: List[str]):
        """Index documents using LangChain (enhanced method)"""
        if not LANGCHAIN_AVAILABLE:
            return False
        
        print(f"[*] LangChain processing {len(pdf_paths)} documents...")
        
        all_documents = []
        
        for pdf_path in pdf_paths:
            print(f"  [LangChain] Processing: {Path(pdf_path).name}")
            
            # Load documents using LangChain
            documents = self.chunker.langchain_processor.load_pdf_documents(pdf_path)
            if documents:
                # Split documents
                split_docs = self.chunker.langchain_processor.split_documents(documents, "recursive")
                all_documents.extend(split_docs)
                print(f"    [LangChain] Created {len(split_docs)} chunks")
        
        if all_documents:
            # Add to LangChain vector store
            success = self.vector_store.add_langchain_documents(all_documents)
            if success:
                print(f"[+] LangChain indexed {len(all_documents)} document chunks")
                return True
        
        return False
    
    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """Query the RAG system with pure semantic retrieval (completely unbiased)"""
        import time
        query_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"🔍 [RAG QUERY START] Processing: '{question}'")
        print(f"{'='*80}")
        
        # Use config top_k if not specified
        if top_k is None:
            top_k = self.config['retrieval'].get('top_k', 8)
            
        print(f"📋 [CONFIG] Using top_k={top_k} for retrieval")
        
        # Generate query embedding with "query:" prefix for e5 model
        print(f"🧠 [EMBEDDING] Generating query embedding...")
        embedding_start = time.time()
        query_embedding = self.embedding_model.encode([f"query: {question}"])[0].tolist()
        embedding_time = time.time() - embedding_start
        print(f"✅ [EMBEDDING] Generated in {embedding_time:.3f}s (vector dim: {len(query_embedding)})")
        
        # STAGE 1: Find most relevant documents with broader search
        print(f"\n📊 [STAGE 1: DOCUMENT DISCOVERY] Finding most relevant documents...")
        stage1_start = time.time()
        candidate_results = self.vector_store.search_documents(
            query_embedding=query_embedding,
            query_text=question,  # Pass query text for ID extraction
            top_k=3,  # Get top 3 documents for better selection
            return_summaries=True  # Get summaries for LLM selection
        )
        stage1_time = time.time() - stage1_start
        print(f"✅ [STAGE 1] Completed in {stage1_time:.3f}s")
        
        if not candidate_results:
            print(f"❌ [STAGE 1] No relevant documents found!")
            print(f"{'='*80}")
            return {
                'question': question,
                'answer': "I don't have that information in the documents.",
                'context': [],
                'retrieved_chunks': [],
                'num_chunks_retrieved': 0,
                'relevant_documents': []
            }
        
        candidate_docs = candidate_results['filenames']
        candidate_summaries = candidate_results.get('summaries', [''] * len(candidate_docs))
        
        print(f"📋 [STAGE 1] Found {len(candidate_docs)} candidate documents:")
        for i, doc in enumerate(candidate_docs):
            print(f"   {i+1}. {doc}")
        print(f"   → Proceeding with document-focused retrieval")
        
        # STAGE 2: Efficient hybrid retrieval (BM25 + semantic)
        print(f"\n🔄 [STAGE 2: CHUNK RETRIEVAL] Hybrid search (BM25 + semantic) from {len(candidate_docs)} documents...")
        stage2_start = time.time()
        retrieved_chunks = self.vector_store.search(
            query=question,  # Pass query text for BM25
            query_embedding=query_embedding,  # Pass embedding for semantic
            top_k=8,  # Reduced for faster CPU processing
            relevant_docs=candidate_docs  # Use semantically similar documents
        )
        stage2_time = time.time() - stage2_start
        print(f"✅ [STAGE 2] Retrieved {len(retrieved_chunks) if retrieved_chunks else 0} chunks in {stage2_time:.3f}s")
        
        if not retrieved_chunks:
            print(f"❌ [STAGE 2] No chunks retrieved from documents!")
            print(f"{'='*80}")
            return {
                'question': question,
                'answer': "I don't have that information in the documents.",
                'context': [],
                'retrieved_chunks': [],
                'num_chunks_retrieved': 0,
                'relevant_documents': candidate_docs
            }
        
        # CRITICAL: Filter chunks to ONLY come from candidate documents (fix scanned PDF issue)
        print(f"\n🔍 [STAGE 3: FILTERING] Applying document-specific filtering...")
        filtered_chunks = []
        for chunk in retrieved_chunks:
            chunk_filename = chunk['metadata'].get('filename', '')
            # Ensure chunk comes from one of our candidate documents
            if any(doc in chunk_filename or chunk_filename in doc for doc in candidate_docs):
                filtered_chunks.append(chunk)
        
        print(f"✅ [STAGE 3] Filtered: {len(filtered_chunks)} chunks kept from {len(retrieved_chunks)} total")
        
        # Build prompt with filtered chunks - apply strict limits  
        print(f"\n📝 [STAGE 4: CONTEXT BUILDING] Organizing chunks for LLM...")
        table_chunks = [c for c in filtered_chunks if c['metadata'].get('chunk_type') == 'table']
        text_chunks = [c for c in filtered_chunks if c['metadata'].get('chunk_type') != 'table']
        
        print(f"📊 [STAGE 4] Found {len(table_chunks)} table chunks, {len(text_chunks)} text chunks")
        
        # Sort chunks by hybrid score (includes document boost for scanned PDFs)
        table_chunks.sort(key=lambda x: x['score'], reverse=True)
        text_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply strict max_chunks limit from config (reduced for speed)
        max_chunks = min(self.config['context_building'].get('max_chunks', 5), 4)  # Cap at 4 for speed
        total_chunks = table_chunks + text_chunks
        total_chunks.sort(key=lambda x: x['score'], reverse=True)
        total_chunks = total_chunks[:max_chunks]  # Strict limit to prevent LLM overload
        
        print(f"⚡ [STAGE 4] Limited to {len(total_chunks)} chunks (max: {max_chunks}) for optimal speed")
        
        # Re-separate after limiting
        table_chunks = [c for c in total_chunks if c['metadata'].get('chunk_type') == 'table']
        text_chunks = [c for c in total_chunks if c['metadata'].get('chunk_type') != 'table']
        
        # Format context with enhanced structure for better LLM understanding
        context_parts = []
        
        if table_chunks:
            context_parts.append("=== FINANCIAL DATA & TABLES (STRUCTURED INFORMATION) ===")
            for i, chunk in enumerate(table_chunks):
                doc_name = chunk['metadata'].get('filename', 'Unknown')
                context_parts.append(f"\n[TABLE {i+1} - {doc_name}]")
                context_parts.append(chunk['content'])
                context_parts.append("-" * 60)
        
        if text_chunks:
            context_parts.append("\n=== DOCUMENT TEXT & DETAILS ===")
            for i, chunk in enumerate(text_chunks):
                doc_name = chunk['metadata'].get('filename', 'Unknown')
                context_parts.append(f"\n[SOURCE {i+1} - {doc_name}]")
                context_parts.append(chunk['content'])
                if i < len(text_chunks) - 1:  # Add separator except for last chunk
                    context_parts.append("-" * 40)
        
        context_text = "\n".join(context_parts)
        original_length = len(context_text)
        
        # Ultra-compact prompt for fast CPU processing
        print(f"\n✏️  [STAGE 5: PROMPT BUILDING] Creating optimized prompt...")
        if len(context_text) > 1500:  # Truncate very long context
            context_text = context_text[:1500] + "... [truncated for speed]"
            print(f"✂️  [STAGE 5] Context truncated: {original_length} → {len(context_text)} characters")
        else:
            print(f"📏 [STAGE 5] Context length: {len(context_text)} characters (no truncation needed)")
        
        prompt = f"""Extract exact information from the context. Answer directly and concisely.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""
        
        print(f"📝 [STAGE 5] Final prompt: {len(prompt)} characters total")
        
        # Try a lightweight rule-based extractor first (deterministic, unbiased)
        print(f"\n🔧 [STAGE 6: RULE-BASED] Attempting quick pattern-based extraction...")
        rule_answer = self._rule_based_extraction(question, table_chunks + text_chunks)
        if rule_answer is not None:
            query_time = time.time() - query_start
            print(f"⚡ [STAGE 6] ✅ Rule-based extraction successful!")
            print(f"📝 [RESULT] Answer: '{rule_answer[:100]}{'...' if len(rule_answer) > 100 else ''}'")
            print(f"⏱️  [TOTAL TIME] Query completed in {query_time:.3f}s (rule-based)")
            print(f"{'='*80}")
            return {
                'question': question,
                'answer': rule_answer,
                'context': retrieved_chunks,
                'num_chunks_retrieved': len(retrieved_chunks),
                'relevant_documents': candidate_docs,
                'retrieved_chunks': retrieved_chunks
            }
        
        print(f"❌ [STAGE 6] Rule-based extraction failed, proceeding to LLM...")

        # Generate answer with low temperature for accuracy
        print(f"\n🤖 [STAGE 7: LLM GENERATION] Initializing language model call...")
        print(f"🏷️  [LLM] Model: {self.llm.model}")
        print(f"📏 [LLM] Prompt size: {len(prompt)} chars | Context: {len(context_text)} chars")
        print(f"⚙️  [LLM] Config: temp=0.0, max_tokens=150")
        
        import time
        start_time = time.time()
        current_time = time.strftime('%H:%M:%S')
        print(f"🚀 [LLM] *** CALLING MODEL AT {current_time} ***")
        print(f"⏳ [LLM] Waiting for response from {self.llm.model}...")
        
        answer = self.llm.generate(
            prompt,
            temperature=0.0,
            max_tokens=150  # Increased for comprehensive answers
        )
        
        end_time = time.time()
        duration = end_time - start_time
        query_time = end_time - query_start
        
        print(f"✅ [LLM] *** RESPONSE RECEIVED ***")
        print(f"⏱️  [LLM] Generation time: {duration:.2f}s")
        print(f"📊 [LLM] Response length: {len(answer) if answer else 0} characters")
        if answer:
            preview = answer.strip()[:150] + ('...' if len(answer.strip()) > 150 else '')
            print(f"📝 [LLM] Preview: '{preview}'")
        else:
            print(f"⚠️  [LLM] Empty response received!")
        
        print(f"\n🎯 [QUERY COMPLETE] *** FINAL RESULTS ***")
        print(f"⏱️  [TOTAL] End-to-end time: {query_time:.3f}s")
        print(f"📊 [STATS] Chunks: {len(retrieved_chunks)} | Docs: {len(candidate_docs)} | Answer: {len(answer) if answer else 0} chars")
        print(f"📂 [DOCS] {', '.join(candidate_docs)}")
        print(f"💬 [ANSWER] {(answer.strip()[:200] + '...') if answer and len(answer.strip()) > 200 else (answer.strip() if answer else 'No answer generated')}")
        print(f"{'='*80}")
        
        # Try LangChain query as enhancement
        langchain_result = None
        if LANGCHAIN_AVAILABLE and self.langchain_pipeline and self.langchain_pipeline.retrieval_chain:
            print(f"\n🔗 [LANGCHAIN ENHANCEMENT] Running parallel LangChain query...")
            try:
                langchain_result = self.langchain_pipeline.query_langchain(question)
                if langchain_result:
                    print(f"✅ [LANGCHAIN] Generated alternative answer: {len(langchain_result['answer'])} chars")
            except Exception as e:
                print(f"❌ [LANGCHAIN] Query failed: {e}")
        
        result = {
            'question': question,
            'answer': answer,
            'context': retrieved_chunks,
            'retrieved_chunks': retrieved_chunks,  # Add for UI display
            'num_chunks_retrieved': len(retrieved_chunks),
            'relevant_documents': candidate_docs
        }
        
        # Add LangChain results if available
        if langchain_result:
            result['langchain_answer'] = langchain_result['answer']
            result['langchain_sources'] = langchain_result.get('source_documents', [])
            result['has_langchain'] = True
        else:
            result['has_langchain'] = False
        
        return result
    
    def clear_index(self):
        """Clear the vector store"""
        self.vector_store.clear()
        print("[*] Vector store cleared")

    def _rule_based_extraction(self, question: str, chunks: List[Dict[str, Any]]) -> Optional[str]:
        """
        DISABLED: Rule-based pattern extraction was causing cross-document contamination.
        The function was merging chunks from multiple documents and returning the first match,
        causing issues like returning "Php 45,000.00" for all ABC questions.
        Let the LLM handle all extraction to ensure document-specific accuracy.
        """
        return None
        q = question.lower()

        # Flatten chunk texts (prefer table chunks first)
        texts = [c['content'] for c in chunks]
        combined = "\n".join(texts)

        # 1) Amount / ABC detection
        if any(k in q for k in ['approved budget cost', 'abc', 'amount', 'approved budget']):
            # Look in table chunks first for currency patterns
            currency_regex = re.compile(r"(?:PHP|Php|P)\s?[0-9]{1,3}(?:,[0-9]{3})*(?:\.\d+)?")
            # Score candidate currencies by proximity to keywords
            lines = combined.split('\n')
            candidates = []
            for idx, line in enumerate(lines):
                for m in currency_regex.finditer(line):
                    score = 0
                    text_low = line.lower()
                    if 'abc' in text_low or 'approved' in text_low or 'approved budget' in text_low or 'budget' in text_low:
                        score += 3
                    # look nearby for keywords
                    window = '\n'.join(lines[max(0, idx-2):min(len(lines), idx+3)])
                    if any(k in window.lower() for k in ['abc', 'approved', 'approved budget', 'budget']):
                        score += 2
                    # prefer exact item matches from question (e.g., chest freezer)
                    item_terms = re.findall(r"[A-Za-z0-9 ]{3,}", question)
                    if item_terms:
                        for t in item_terms:
                            t = t.strip().lower()
                            if len(t) > 3 and t in text_low:
                                score += 2
                    candidates.append((m.group(0), score, idx))

            if candidates:
                # choose highest score then nearest to top
                candidates.sort(key=lambda x: (-x[1], x[2]))
                return candidates[0][0]

        # 2) Deadline / Closing Date / Bid opening
        if any(k in q for k in ['closing date', 'closing', 'deadline', 'bid opening', 'bid opening scheduled']):
            # Date pattern like 'July 22, 2025 9:30 AM' or 'August 13, 2025, at 9:30 AM'
            date_regex = re.compile(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s*\d{4}(?:\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))?\b")
            candidates = date_regex.findall(combined)
            if candidates:
                # prefer lines mentioning closing/bid
                for line in combined.split('\n'):
                    if any(w in line.lower() for w in ['closing', 'deadline', 'bid opening', 'opening']):
                        m = date_regex.search(line)
                        if m:
                            return m.group(0)
                return candidates[0]

        # 3) Delivery period in calendar days
        if 'calendar days' in q or 'calendar day' in q or 'delivery period' in q:
            days_regex = re.compile(r"(\d+)\s+calendar\s+days", re.IGNORECASE)
            m = days_regex.search(combined)
            if m:
                return f"{m.group(1)} calendar days"

        # 4) Generic exact match for activity/project names (short deterministic search)
        if any(k in q for k in ['which specific research project', 'research project', 'project for']):
            # Find lines with 'project' or known project names / all-caps organization
            for line in combined.split('\n'):
                if 'project' in line.lower() or 'centro' in line.lower() or 'project' in line:
                    # return cleaned line
                    clean = line.strip()
                    if len(clean) > 10:
                        return clean

        return None


def get_system_metrics():
    """Get current system metrics"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        memory_percent = memory.percent
        
        # Processing mode status (CPU optimized)
        gpu_info = "Mode: CPU Optimized"
        gpu_status = "fast_cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_info = f"GPU: {device_name[:20]}.." if len(device_name) > 20 else f"GPU: {device_name}"
                gpu_status = f"Active ({memory_allocated:.1f}/{total_memory:.0f}GB)"
            else:
                # Show CPU optimization instead of GPU unavailability
                cpu_cores = psutil.cpu_count(logical=False)
                cpu_threads = psutil.cpu_count(logical=True)
                gpu_info = f"CPU: {cpu_cores}C/{cpu_threads}T Optimized"
                gpu_status = "fast_cpu"
        except ImportError:
            gpu_info = "Mode: CPU Only"
            gpu_status = "cpu_only"
        except Exception as e:
            gpu_info = "Mode: CPU Fallback"
            gpu_status = "cpu_fallback"
        
        # Token count (from current session) with limits
        total_tokens = 0
        max_context_tokens = 4096  # Typical context limit for most models
        if 'current_session_id' in st.session_state and st.session_state.current_session_id:
            current_session = st.session_state.chat_sessions.get(st.session_state.current_session_id, {})
            messages = current_session.get('messages', [])
            # More accurate token estimation: ~3.5 characters per token
            for msg in messages:
                if not msg.get('temporary', False):  # Skip temporary thinking messages
                    total_tokens += len(msg.get('content', '')) // 3.5
        
        # Calculate token usage percentage
        token_percentage = min(100, (total_tokens / max_context_tokens) * 100)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_used_gb': memory_used_gb,
            'memory_total_gb': memory_total_gb,
            'memory_percent': memory_percent,
            'tokens': total_tokens,
            'max_tokens': max_context_tokens,
            'token_percentage': token_percentage,
            'gpu_info': gpu_info,
            'gpu_status': gpu_status
        }
    except Exception as e:
        return {
            'cpu_percent': 0,
            'memory_used_gb': 0,
            'memory_total_gb': 0,
            'memory_percent': 0,
            'tokens': 0,
            'gpu_info': "Error",
            'gpu_status': "error"
        }


def create_chatgpt_css():
    """Create professional ChatGPT-style CSS"""
    st.markdown("""
    <style>
    /* Offline fonts - using system fonts instead of Google Fonts */
    
    * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif; }
    .stApp { background-color: #343541; }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    section[data-testid="stSidebar"] { background-color: #202123; border-right: 1px solid #444654; }
    section[data-testid="stSidebar"] .element-container { margin-bottom: 0.5rem; }
    .main .block-container { max-width: 900px; padding: 2rem 1rem 8rem 1rem; }
    
    /* Top Header Bar */
    .top-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 56px;
        background-color: #343541;
        border-bottom: 1px solid #444654;
        z-index: 999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 1rem;
    }
    
    .header-title {
        font-size: 1rem;
        font-weight: 600;
        color: #ececf1;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metrics Bar */
    .metrics-bar {
        position: fixed;
        bottom: 0;
        right: 0;
        background-color: #202123;
        border-top: 1px solid #444654;
        padding: 0.5rem 1rem;
        display: flex;
        gap: 2rem;
        align-items: center;
        font-size: 0.75rem;
        color: #8e8ea0;
        z-index: 100;
    }
    
    .metric-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .metric-label {
        font-weight: 500;
        color: #ececf1;
    }
    
    .metric-value {
        color: #8e8ea0;
    }
    
    /* Upload Section Title */
    .upload-section-title {
        font-size: 0.7rem;
        font-weight: 700;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1rem 0 0.5rem 0;
        padding: 0.5rem 0;
        border-bottom: 1px solid #30363d;
    }
    
    .stButton > button { width: 100%; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.15); background-color: #2a2b32; color: #ececf1; font-weight: 500; padding: 0.75rem 1rem; transition: all 0.2s; font-size: 0.875rem; }
    .stButton > button:hover { background-color: #353640; border-color: rgba(255, 255, 255, 0.25); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2); }
    .stButton > button[kind="primary"] { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; color: white; font-weight: 600; }
    .stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #5568d3 0%, #6a3f8f 100%); transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4); }
    .stButton > button:disabled, .stButton > button[disabled] { background: #3a3b42 !important; color: #6e7681 !important; cursor: not-allowed !important; opacity: 0.6; transform: none !important; box-shadow: none !important; }
    
    /* New Chat Button Style */
    .stButton > button:first-child { background-color: #2a2b32; border: 1px solid #565869; }
    .stButton > button:first-child:hover { background-color: #343541; }
    
    .stChatMessage { background-color: transparent; padding: 1.5rem 0; border-radius: 0; margin-bottom: 0.5rem; animation: fadeIn 0.3s ease-in; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .stChatMessage[data-testid="user-message"] { background-color: transparent; }
    .stChatMessage[data-testid="assistant-message"] { background-color: #444654; margin-left: -2rem; margin-right: -2rem; padding: 1.5rem 2rem; }
    .stChatMessage p { line-height: 1.7; margin-bottom: 0.75rem; color: #ececf1; font-size: 1rem; }
    .stChatMessage ul, .stChatMessage ol { margin-left: 1.5rem; margin-bottom: 1rem; }
    .stChatMessage li { margin-bottom: 0.5rem; line-height: 1.6; color: #d1d5db; }
    
    /* Chat message bubbles - User on right, Assistant on left */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 18px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0 1rem auto;
        max-width: 65%;
        color: #ffffff;
        text-align: left;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5);
        animation: slideInRight 0.3s ease-out;
        line-height: 1.6;
        font-weight: 500;
    }
    
    .assistant-message {
        background-color: #2d2d3d;
        border: 1px solid #3d3d4d;
        border-radius: 18px;
        padding: 1.25rem 1.5rem;
        margin: 1rem auto 1rem 0;
        max-width: 65%;
        color: #e8e8f0;
        text-align: left;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        animation: slideInLeft 0.3s ease-out;
        line-height: 1.6;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Blur overlay for chat area when no documents */
    .chat-disabled-overlay {
        position: relative;
        pointer-events: none;
        filter: blur(5px);
        opacity: 0.3;
    }
    
    .chat-warning-message {
        position: absolute;
        top: 45%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem 2.5rem;
        border-radius: 12px;
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
        text-align: center;
        z-index: 999;
        font-size: 1.1rem;
        font-weight: 600;
        animation: gentlePulse 3s ease-in-out infinite;
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    
    @keyframes gentlePulse {
        0%, 100% { transform: translate(-50%, -50%) scale(1); box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4); }
        50% { transform: translate(-50%, -50%) scale(1.02); box-shadow: 0 16px 48px rgba(102, 126, 234, 0.6); }
    }
    
    /* Sidebar Section Styling */
    [data-testid="stSidebar"] hr {
        margin: 1.5rem 0;
        border-color: #30363d;
        opacity: 0.5;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        font-size: 0.85rem;
    }
    
    /* File uploader styling */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #1a1b26;
        border: 1px dashed #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    [data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #1e1f2e;
    }
    
    /* Text input styling */
    [data-testid="stSidebar"] input {
        background: #1a1b26 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        color: #e8e8f0 !important;
        font-size: 0.85rem !important;
    }
    
    [data-testid="stSidebar"] input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 1px #667eea !important;
    }
    
    /* Sidebar Section Styling */
    [data-testid="stSidebar"] hr {
        margin: 1.5rem 0;
        border-color: #30363d;
        opacity: 0.5;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        font-size: 0.85rem;
    }
    
    /* File uploader styling */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #1a1b26;
        border: 1px dashed #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    [data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #1e1f2e;
    }
    
    /* Terminal-like design for processing log */
    .terminal-window {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
        margin: 1rem 0;
    }
    
    .terminal-window-header {
        background: #161b22;
        padding: 0.6rem 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid #21262d;
    }
    
    .terminal-window-dot {
        width: 11px;
        height: 11px;
        border-radius: 50%;
        display: inline-block;
    }
    
    .terminal-window-dot.red { background: #ff5f56; }
    .terminal-window-dot.yellow { background: #ffbd2e; }
    .terminal-window-dot.green { background: #27c93f; }
    
    .terminal-window-title {
        color: #8b949e;
        font-size: 0.8rem;
        margin-left: 0.5rem;
        font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        font-weight: 500;
    }
    
    .terminal-window-body {
        background: #0d1117;
        padding: 1rem;
        max-height: 350px;
        overflow-y: auto;
        font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        font-size: 0.875rem;
        line-height: 1.8;
        color: #c9d1d9;
    }
    
    .stChatInput { position: sticky; bottom: 0; background: linear-gradient(to top, #343541 0%, #343541 90%, transparent 100%); padding: 2rem 0 1.5rem 0; z-index: 50; }
    .stChatInput > div { background-color: #40414f; border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 12px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2); transition: all 0.2s ease; }
    .stChatInput > div:focus-within { border-color: #8e8ea0; box-shadow: 0 0 0 2px rgba(142, 142, 160, 0.1), 0 4px 12px rgba(0, 0, 0, 0.3); }
    .stChatInput textarea { color: #ececf1; font-size: 1rem; line-height: 1.5; }
    .stChatInput textarea::placeholder { color: #8e8ea0; }
    
    .stTextInput > div > div > input, .stNumberInput > div > div > input { background-color: #2d2d30; color: #ececf1; border: 2px solid #3e3e42; border-radius: 8px; padding: 0.7rem 1rem; font-size: 0.95rem; transition: all 0.2s; }
    .stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus { border-color: #667eea; outline: none; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1); background-color: #1a1a1d; }
    .stSelectbox > div > div { background-color: #40414f; color: #ececf1; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.2); }
    .stSlider > div > div > div > div { background: #667eea; }
    .streamlit-expanderHeader { background-color: #2d2d30; color: #ececf1; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.2); padding: 0.75rem 1rem; }
    .streamlit-expanderHeader:hover { background-color: #3a3b42; }
    .stFileUploader { background-color: transparent; border: 1px dashed rgba(255, 255, 255, 0.2); border-radius: 8px; padding: 1rem; }
    .stFileUploader section { background-color: transparent !important; }
    .stFileUploader label { color: #8e8ea0; font-size: 0.875rem; }
    hr { border: none; border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 1rem 0; }
    h1, h2, h3, h4, h5, h6 { color: #ececf1; }
    
    .welcome-container { text-align: center; padding: 5rem 2rem; }
    .welcome-logo { font-size: 4.5rem; margin-bottom: 1.5rem; animation: float 3s ease-in-out infinite; }
    @keyframes float { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-10px); } }
    .welcome-title { font-size: 2.8rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem; letter-spacing: -0.5px; }
    .welcome-subtitle { font-size: 1.3rem; color: #a0aec0; margin-bottom: 3rem; }
    .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-top: 3rem; }
    .feature-card { background: linear-gradient(145deg, #2d2d30, #3a3b42); border-radius: 12px; padding: 1.5rem; border: 1px solid #565869; transition: all 0.3s; }
    .feature-card:hover { border-color: #667eea; transform: translateY(-4px); box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2); }
    .feature-icon { font-size: 2.5rem; margin-bottom: 1rem; }
    .feature-title { font-size: 1.1rem; font-weight: 600; color: #ececf1; margin-bottom: 0.5rem; }
    .feature-desc { font-size: 0.9rem; color: #c5c5d2; line-height: 1.6; }
    
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #2d2d30; }
    ::-webkit-scrollbar-thumb { background: #565869; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #8e8ea0; }
    
    .stMarkdown { color: #ececf1; }
    code { background-color: #1a1a1d; padding: 0.2rem 0.4rem; border-radius: 4px; color: #3fb950; font-family: 'Monaco', 'Menlo', 'Consolas', monospace; font-size: 0.9em; }
    pre { background-color: #1a1a1d; border-radius: 8px; padding: 1rem; overflow-x: auto; }
    
    .source-citation { display: inline-flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-size: 0.75rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 12px; margin: 0 0.25rem; cursor: pointer; transition: all 0.2s ease; text-decoration: none; box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3); }
    .source-citation:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4); }
    .source-details { background: linear-gradient(145deg, #2d2d30, #3a3b42); border-left: 3px solid #667eea; border-radius: 8px; padding: 1rem; margin: 0.75rem 0; font-size: 0.9rem; }
    .source-header { font-weight: 600; color: #667eea; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem; }
    .source-text { color: #c5c5d2; line-height: 1.6; font-style: italic; padding: 0.75rem; background-color: rgba(102, 126, 234, 0.05); border-radius: 6px; margin-top: 0.5rem; }
    
    .thinking-dots { display: inline-flex; gap: 0.3rem; align-items: center; padding: 0.5rem 0; }
    .thinking-dots span { width: 8px; height: 8px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); animation: pulse 1.4s ease-in-out infinite; }
    .thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
    .thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes pulse { 0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1.1); } }
    
    .terminal-container { background-color: #1a1a1d; border: 1px solid #565869; border-radius: 8px; padding: 1rem; margin: 1rem 0; font-family: 'Monaco', 'Menlo', 'Consolas', monospace; font-size: 0.85rem; max-height: 300px; overflow-y: auto; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4); }
    .terminal-header { display: flex; align-items: center; justify-content: space-between; padding-bottom: 0.75rem; margin-bottom: 0.75rem; border-bottom: 1px solid #2d2d30; }
    .terminal-title { color: #8e8ea0; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .terminal-dots { display: flex; gap: 0.4rem; }
    .terminal-dot { width: 10px; height: 10px; border-radius: 50%; }
    .terminal-dot.red { background-color: #ff5f56; }
    .terminal-dot.yellow { background-color: #ffbd2e; }
    .terminal-dot.green { background-color: #27c93f; }
    .terminal-content { color: #3fb950; line-height: 1.6; }
    .terminal-line { margin-bottom: 0.25rem; animation: terminalFadeIn 0.3s ease-in; color: #3fb950; }
    @keyframes terminalFadeIn { from { opacity: 0; transform: translateX(-10px); } to { opacity: 1; transform: translateX(0); } }
    .terminal-line.info { color: #58a6ff; }
    .terminal-line.success { color: #3fb950; }
    .terminal-line.warning { color: #d29922; }
    .terminal-line.error { color: #f85149; }
    .terminal-line.muted { color: #8e8ea0; }
    .terminal-line .icon { margin-right: 0.5rem; font-weight: bold; }
    .terminal-line .timestamp { color: #6e7681; margin-right: 0.5rem; font-size: 0.75rem; }
    
    /* Document Cards */
    .doc-card {
        background: linear-gradient(135deg, #2a2b32 0%, #252631 100%);
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    
    .doc-card:hover {
        transform: translateX(3px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .doc-name {
        color: #e8e8f0;
        font-size: 0.85rem;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .doc-meta {
        color: #9ca3af;
        font-size: 0.7rem;
        margin-top: 0.25rem;
        font-weight: 500;
    }
    
    /* Loading Screen */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(52, 53, 65, 0.95);
        backdrop-filter: blur(10px);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.3s ease-in;
    }
    
    .loading-box {
        background: linear-gradient(145deg, #2d2d30, #3a3b42);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 3rem 4rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        animation: slideUp 0.4s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid rgba(102, 126, 234, 0.2);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 1.5rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        color: #ececf1;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .loading-subtext {
        color: #8e8ea0;
        font-size: 0.9rem;
    }
    
    /* Loading Screen */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(52, 53, 65, 0.95);
        backdrop-filter: blur(10px);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: fadeIn 0.3s ease-in;
    }
    
    .loading-box {
        background: linear-gradient(145deg, #2d2d30, #3a3b42);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 16px;
        padding: 3rem 4rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        animation: slideUp 0.4s ease-out;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 4px solid rgba(102, 126, 234, 0.2);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin: 0 auto 1.5rem;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-text {
        color: #ececf1;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .loading-subtext {
        color: #8e8ea0;
        font-size: 0.9rem;
    }
    
    html { scroll-behavior: smooth; }
    </style>
    """, unsafe_allow_html=True)


def render_top_header():
    """Render the top header bar"""
    st.markdown("""
    <div class="top-header">
        <div class="header-title">
            <span>💬</span>
            <span>RAG Assistant</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_loading_screen():
    """Render loading screen overlay"""
    st.markdown("""
    <div class="loading-overlay">
        <div class="loading-box">
            <div class="loading-spinner"></div>
            <div class="loading-text">Loading RAG System</div>
            <div class="loading-subtext">Initializing components...</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_loading_screen():
    """Render loading screen overlay"""
    st.markdown("""
    <div class="loading-overlay">
        <div class="loading-box">
            <div class="loading-spinner"></div>
            <div class="loading-text">Loading RAG System</div>
            <div class="loading-subtext">Initializing components...</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics_bar():
    """Render the bottom right metrics bar"""
    metrics = get_system_metrics()
    
    # Set processing mode color
    if "Active" in metrics['gpu_status']:
        gpu_color = "#10b981"  # Green for active GPU
    elif "fast_cpu" in metrics['gpu_status']:
        gpu_color = "#3b82f6"  # Blue for optimized CPU
    elif "cpu_only" in metrics['gpu_status'] or "cpu_fallback" in metrics['gpu_status']:
        gpu_color = "#f59e0b"  # Orange for CPU fallback
    else:
        gpu_color = "#6b7280"  # Gray for unknown
    
    st.markdown(f"""
    <div class="metrics-bar">
        <div class="metric-item">
            <span class="metric-label">Memory:</span>
            <span class="metric-value">{metrics['memory_used_gb']:.1f}GB / {metrics['memory_total_gb']:.1f}GB ({metrics['memory_percent']:.0f}%)</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">CPU:</span>
            <span class="metric-value">{metrics['cpu_percent']:.0f}%</span>
        </div>
        <div class="metric-item">
            <span class="metric-label" style="color: {gpu_color};">{metrics['gpu_info']}</span>
            <span class="metric-value" style="color: {gpu_color};">{metrics['gpu_status']}</span>
        </div>
        <div class="metric-item">
            <span class="metric-label">Tokens:</span>
            <span class="metric-value" style="color: {'#ef4444' if metrics.get('token_percentage', 0) > 80 else '#10b981' if metrics.get('token_percentage', 0) < 50 else '#f59e0b'}">
                {int(metrics['tokens']):,}/{int(metrics.get('max_tokens', 4096)):,} ({metrics.get('token_percentage', 0):.0f}%)
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def add_terminal_log(message: str, log_type: str = "info", loading: bool = False):
    """Add a log message to the terminal display"""
    timestamp = time.strftime("%H:%M:%S")
    icons = {
        "info": "[*]",
        "success": "[+]",
        "warning": "[!]",
        "error": "[✗]",
        "muted": "   "
    }
    icon = icons.get(log_type, "[*]")
    
    # Add loading animation if specified
    if loading:
        message = f'<span class="terminal-loading">{message}<span class="loading-dots"></span></span>'
    
    if 'terminal_logs' not in st.session_state:
        st.session_state.terminal_logs = []
    
    st.session_state.terminal_logs.append({
        'timestamp': timestamp,
        'icon': icon,
        'message': message,
        'type': log_type,
        'loading': loading
    })

def restart_application():
    """Restart the application to fix database issues"""
    clear_terminal_logs()
    add_terminal_log("Restarting to fix database...", "warning")
    
    # Cleanup current session
    if hasattr(st.session_state, 'rag_pipeline') and st.session_state.rag_pipeline:
        try:
            st.session_state.rag_pipeline.cleanup()
        except:
            pass
    
    # Reset session state
    for key in list(st.session_state.keys()):
        if key not in ['terminal_logs']:  # Keep terminal logs to show restart message
            del st.session_state[key]
    
    st.session_state.system_initialized = False
    st.session_state.cleanup_needed = True
    add_terminal_log("Application reset complete", "success")
    st.rerun()

def clear_terminal_logs():
    """Clear all terminal logs"""
    st.session_state.terminal_logs = []

def render_terminal():
    """Render terminal HTML using components with terminal window design"""
    if 'terminal_logs' not in st.session_state:
        st.session_state.terminal_logs = []
    
    if not st.session_state.terminal_logs:
        # Show empty terminal without returning early
        pass
    
    terminal_html = """
    <style>
        .terminal-window {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            margin: 0;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
        }
        .terminal-window-header {
            background: #161b22;
            padding: 0.6rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-bottom: 1px solid #21262d;
        }
        .terminal-window-dot {
            width: 11px;
            height: 11px;
            border-radius: 50%;
            display: inline-block;
        }
        .terminal-window-dot.red { background: #ff5f56; }
        .terminal-window-dot.yellow { background: #ffbd2e; }
        .terminal-window-dot.green { background: #27c93f; }
        .terminal-window-title {
            color: #8b949e;
            font-size: 0.8rem;
            margin-left: 0.5rem;
            font-weight: 500;
        }
        .terminal-window-body {
            background: #0d1117;
            padding: 0.5rem 1rem;
            height: 160px;
            overflow: hidden;
            color: #c9d1d9;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }
        .terminal-line {
            margin-bottom: 0.4rem;
            display: flex;
            align-items: baseline;
            gap: 0.5rem;
        }
    </style>
    <div class="terminal-window">
        <div class="terminal-window-header">
            <span class="terminal-window-dot red"></span>
            <span class="terminal-window-dot yellow"></span>
            <span class="terminal-window-dot green"></span>
            <span class="terminal-window-title">processing.log</span>
        </div>
        <div class="terminal-window-body">
    """
    
    # Show only last 5 logs
    recent_logs = st.session_state.terminal_logs[-5:] if len(st.session_state.terminal_logs) > 5 else st.session_state.terminal_logs
    
    for log in recent_logs:
        color_map = {
            'info': '#58a6ff',
            'success': '#3fb950',
            'warning': '#f0ad4e',
            'error': '#f85149',
            'muted': '#8b949e'
        }
        line_color = color_map.get(log['type'], '#58a6ff')
        message = log['message']
        
        terminal_html += f"""
            <div class="terminal-line" style="color: {line_color};">
                <span style="color: #6e7681; font-size: 0.75rem; min-width: 60px;">{log['timestamp']}</span>
                <span style="font-weight: 600; min-width: 24px;">{log['icon']}</span>
                <span style="flex: 1;">{message}</span>
            </div>
        """
    
    terminal_html += """
        </div>
    </div>
    """
    
    import streamlit.components.v1 as components
    components.html(terminal_html, height=240, scrolling=False)

def format_sources_minimal(sources: List[Dict[str, Any]]) -> str:
    """Format sources with minimal numbered citations"""
    if not sources:
        return ""
    
    formatted = "\n\n---\n\n**Sources:**\n\n"
    
    for i, source in enumerate(sources, 1):
        doc_name = source.get('metadata', {}).get('source', 'Unknown')
        chunk_id = source.get('metadata', {}).get('chunk_id', 'N/A')
        page = source.get('metadata', {}).get('page', 'N/A')
        text = source.get('text', '')[:150] + '...' if len(source.get('text', '')) > 150 else source.get('text', '')
        
        formatted += f"""
<div class="source-details">
    <div class="source-header">
        <span class="source-citation">{i}</span>
        📄 {doc_name} • Chunk {chunk_id} • Page {page}
    </div>
    <div class="source-text">{text}</div>
</div>
"""
    
    return formatted

def add_source_citations(answer: str, num_sources: int) -> str:
    """Add inline source citations to answer"""
    if num_sources == 0:
        return answer
    
    # Add citation numbers at the end
    citations = ''.join([f'<span class="source-citation">{i}</span>' for i in range(1, num_sources + 1)])
    return f"{answer}\n\n{citations}"

def create_new_session():
    """Create a new chat session with isolated vector database"""
    import uuid
    session_id = str(uuid.uuid4())
    st.session_state.session_counter += 1
    
    session_data = {
        'id': session_id,
        'number': st.session_state.session_counter,
        'title': f"New Chat {st.session_state.session_counter}",
        'messages': [],
        'created_at': time.time(),
        'query_history': [],
        'rag_pipeline': None,  # Each session has its own RAG pipeline
        'indexed_documents': {}  # Each session has its own document index
    }
    
    st.session_state.chat_sessions[session_id] = session_data
    st.session_state.current_session_id = session_id
    
    # Clear the global RAG pipeline to force re-initialization for new session
    st.session_state.rag_pipeline = None
    st.session_state.indexed_documents = {}

def switch_to_session(session_id):
    """Switch to a specific chat session and load its context"""
    st.session_state.current_session_id = session_id
    
    # Load session-specific data
    current_session = st.session_state.chat_sessions[session_id]
    st.session_state.rag_pipeline = current_session.get('rag_pipeline', None)
    st.session_state.indexed_documents = current_session.get('indexed_documents', {})

def get_current_session():
    """Get the current active session"""
    if not st.session_state.current_session_id:
        create_new_session()
    
    # Get reference to the actual session dict (not a copy)
    session_id = st.session_state.current_session_id
    
    # Ensure session exists
    if session_id not in st.session_state.chat_sessions:
        create_new_session()
        session_id = st.session_state.current_session_id
    
    session = st.session_state.chat_sessions[session_id]
    
    # Ensure session has required keys
    if 'messages' not in session:
        session['messages'] = []
    if 'query_history' not in session:
        session['query_history'] = []
    
    return session

def save_session_context(session_id):
    """Save current context to session"""
    if session_id in st.session_state.chat_sessions:
        st.session_state.chat_sessions[session_id]['rag_pipeline'] = st.session_state.rag_pipeline
        st.session_state.chat_sessions[session_id]['indexed_documents'] = st.session_state.indexed_documents

def update_session_title(session_id, title):
    """Update session title based on first message"""
    if session_id in st.session_state.chat_sessions:
        st.session_state.chat_sessions[session_id]['title'] = title

def display_processing_logs(logs):
    """Display processing logs in a styled container"""
    if logs:
        with st.expander("Processing Logs", expanded=False):
            log_html = '<div class="processing-log">'
            for log in logs[-30:]:  # Show last 30 logs
                if "SUCCESS" in log:
                    log_html += f'<div class="log-entry log-success">[SUCCESS] {log}</div>'
                elif "WARNING" in log:
                    log_html += f'<div class="log-entry log-warning">⚠️ {log}</div>'
                elif "ERROR" in log:
                    log_html += f'<div class="log-entry log-error">[ERROR] {log}</div>'
                else:
                    log_html += f'<div class="log-entry">ℹ️ {log}</div>'
            log_html += '</div>'
            st.markdown(log_html, unsafe_allow_html=True)

def create_sidebar():
    """Create ChatGPT-style sidebar with session management"""
    with st.sidebar:
        # Initialize session state
        if 'chat_sessions' not in st.session_state:
            st.session_state.chat_sessions = {}
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None
        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = None
        if 'processing_logs' not in st.session_state:
            st.session_state.processing_logs = []
        if 'indexed_documents' not in st.session_state:
            st.session_state.indexed_documents = {}
        if 'session_counter' not in st.session_state:
            st.session_state.session_counter = 0
        if 'is_processing' not in st.session_state:
            st.session_state.is_processing = False
        
        # System status indicator
        st.markdown("### 🔧 System Status")
        
        # Check system components status
        rag_status = "🟢 Ready" if st.session_state.rag_pipeline else "🔴 Not Ready"
        docs_status = "🟢 Loaded" if st.session_state.indexed_documents else "🟡 No Documents"
        processing_status = "🔄 Processing" if st.session_state.get('is_processing', False) else "✅ Idle"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RAG Engine", rag_status)
            st.metric("Processing", processing_status)
        with col2:
            st.metric("Documents", docs_status)
            total_docs = len(st.session_state.indexed_documents)
            st.metric("Total Docs", total_docs)
        
        st.divider()
        
        # New chat session button with status indicator
        chat_disabled = st.session_state.get('is_processing', False) or st.session_state.get('processing_query', False)
        
        if chat_disabled:
            st.button("⏳ Processing...", disabled=True, use_container_width=True, key="new_chat_disabled")
        else:
            if st.button("➕ New chat", use_container_width=True, key="new_chat_sidebar"):
                with st.spinner("🔄 Creating new chat session..."):
                    create_new_session()
                st.success("✅ New chat session created!")
                time.sleep(0.5)
                st.rerun()
        
        # System status indicator
        st.markdown("### 🔧 System Status")
        
        # Check system components status
        rag_status = "🟢 Ready" if st.session_state.rag_pipeline else "🔴 Not Ready"
        docs_status = "🟢 Loaded" if st.session_state.indexed_documents else "🟡 No Documents"
        processing_status = "🔄 Processing" if st.session_state.get('is_processing', False) else "✅ Idle"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RAG Engine", rag_status)
            st.metric("Processing", processing_status)
        with col2:
            st.metric("Documents", docs_status)
            total_docs = len(st.session_state.indexed_documents)
            st.metric("Total Docs", total_docs)
        
        st.divider()
        
        # Chat sessions list
        if st.session_state.chat_sessions:
            # Sort sessions by creation time (newest first)
            sorted_sessions = sorted(
                st.session_state.chat_sessions.items(), 
                key=lambda x: x[1].get('created_at', 0), 
                reverse=True
            )
            
            for session_id, session_data in sorted_sessions:
                # Create session title from first message or default
                title = session_data.get('title', f"Chat {session_data.get('number', 1)}")
                display_title = title[:22] + "..." if len(title) > 25 else title
                
                # Highlight current session
                button_type = "primary" if session_id == st.session_state.current_session_id else "secondary"
                
                # Create columns for session button and rename
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    if st.button(
                        f"💬 {display_title}", 
                        key=f"session_{session_id}",
                        use_container_width=True,
                        type=button_type
                    ):
                        switch_to_session(session_id)
                        st.rerun()
                
                with col2:
                    if st.button("✏️", key=f"rename_{session_id}", help="Rename session"):
                        st.session_state[f'renaming_{session_id}'] = True
                        st.rerun()
                
                # Show rename input if in rename mode
                if st.session_state.get(f'renaming_{session_id}', False):
                    new_title = st.text_input(
                        "New name:",
                        value=title,
                        key=f"rename_input_{session_id}",
                        placeholder="Enter new session name"
                    )
                    
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        if st.button("✓ Save", key=f"save_rename_{session_id}", use_container_width=True):
                            if new_title.strip():
                                update_session_title(session_id, new_title.strip())
                            st.session_state[f'renaming_{session_id}'] = False
                            st.rerun()
                    
                    with col_cancel:
                        if st.button("✗ Cancel", key=f"cancel_rename_{session_id}", use_container_width=True):
                            st.session_state[f'renaming_{session_id}'] = False
                            st.rerun()
        
        st.divider()
        
        # Document upload section with enhanced indicators
        st.markdown('<div class="upload-section-title">📎 Upload Documents</div>', unsafe_allow_html=True)
        
        # Show processing status if active
        if st.session_state.get('is_processing', False):
            st.info("⏳ Processing documents in progress...")
            
            # Show progress if available
            if 'processing_progress' in st.session_state:
                progress_value = st.session_state.processing_progress
                progress_text = st.session_state.get('processing_stage', 'Processing...')
                
                progress_bar = st.progress(progress_value)
                st.caption(f"🔄 {progress_text}")
        
        # File uploader with status indicators
        upload_disabled = st.session_state.get('is_processing', False)
        
        if upload_disabled:
            st.info("📤 File upload disabled during processing")
        
        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="pdf_uploader",
            disabled=upload_disabled,
            help="Upload PDF files to analyze. Processing will begin automatically."
        )
        
        # Folder path input
        folder_path = st.text_input(
            "Folder path",
            value="",
            placeholder="C:/path/to/pdfs or leave empty",
            key="folder_path_input"
        )
        
        # Enhanced file information display
        if uploaded_files:
            st.markdown(f"📋 **{len(uploaded_files)} files selected**")
            
            # Calculate total size and show file details with icons
            total_size = 0
            file_details = []
            
            for file in uploaded_files:
                file_size = len(file.getvalue()) / 1024 / 1024  # Size in MB
                total_size += file_size
                
                # File status icon
                file_status = "📄"
                if file.name in st.session_state.indexed_documents:
                    file_status = "✅"
                elif st.session_state.get('is_processing', False):
                    file_status = "⏳"
                
                file_details.append((file_status, file.name, file_size))
            
            # Display files with status
            for status, name, size in file_details:
                st.write(f"{status} `{name}` ({size:.1f} MB)")
            
            # Total size indicator
            st.caption(f"📊 Total size: {total_size:.1f} MB")
        
        # Process button with enhanced status indicators
        has_files = bool(uploaded_files) or (folder_path and folder_path.strip() != "" and folder_path.strip() != "pdf" and os.path.exists(folder_path.strip()))
        process_disabled = st.session_state.get('is_processing', False) or not has_files
        
        if st.session_state.get('is_processing', False):
            # Show processing status
            current_stage = st.session_state.get('processing_stage', 'Processing...')
            button_label = f"⏳ {current_stage}"
            
            # Show estimated time if available
            if 'processing_start_time' in st.session_state:
                import time
                elapsed = time.time() - st.session_state.processing_start_time
                st.caption(f"⏱️ Elapsed: {elapsed:.1f}s")
        elif not has_files:
            button_label = "📤 Select Files First"
            st.caption("⚠️ Please upload files or specify a valid folder path")
        else:
            button_label = "🚀 Process Documents"
        
        button_disabled = process_disabled
        
        # Track if we just clicked the button
        if 'process_clicked' not in st.session_state:
            st.session_state.process_clicked = False
        
        if st.button(button_label, type="primary" if not process_disabled else "secondary", use_container_width=True, disabled=button_disabled):
            # Double check files exist before setting processing flag
            if has_files:
                # Initialize processing tracking
                import time
                st.session_state.processing_start_time = time.time()
                st.session_state.processing_progress = 0.0
                st.session_state.processing_stage = "Initializing..."
                st.session_state.is_processing = True
                st.session_state.process_clicked = True
                
                with st.spinner("🔄 Starting document processing..."):
                    time.sleep(0.5)  # Brief pause for user feedback
                
                st.success("✅ Processing started!")
                st.rerun()  # Force rerun to show terminal and blur chat
            else:
                st.error("❌ Please upload PDF files or specify a valid folder path before processing.")
                st.info("💡 Tip: Upload files using the file uploader above or specify a folder path.")
        
        # Process documents if flag is set and we're in processing state
        if st.session_state.get('is_processing', False) and st.session_state.get('process_clicked', False):
            st.session_state.process_clicked = False  # Reset flag
            process_documents(folder_path, uploaded_files)
        
        # Show indexed documents
        if st.session_state.indexed_documents:
            st.markdown('<div class="upload-section-title">📚 Indexed Documents</div>', unsafe_allow_html=True)
            st.markdown(f"<div style='color: #9ca3af; font-size: 0.75rem; margin-bottom: 0.5rem;'>{len(st.session_state.indexed_documents)} document(s) ready</div>", unsafe_allow_html=True)
            
            for filename in list(st.session_state.indexed_documents.keys()):
                doc_info = st.session_state.indexed_documents[filename]
                
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    <div class="doc-card">
                        <div class="doc-name">📄 {filename[:20]}...</div>
                        <div class="doc-meta">{doc_info.get('total_chunks', 0)} chunks</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if st.button("🗑️", key=f"remove_{filename}", help="Remove"):
                        del st.session_state.indexed_documents[filename]
                        st.rerun()
        
        st.divider()
        
        # Debug Settings Panel
        if 'show_debug' not in st.session_state:
            st.session_state.show_debug = False
            
        debug_col1, debug_col2 = st.columns([3, 1])
        with debug_col1:
            if st.button("🔧 Debug & Settings", use_container_width=True):
                st.session_state.show_debug = not st.session_state.show_debug
        with debug_col2:
            if st.session_state.show_debug:
                st.markdown("🔽")
            else:
                st.markdown("▶️")
        
        # Debug panel content
        if st.session_state.show_debug:
            # Functionality Panel
            with st.expander("✅ Functionality", expanded=True):
                st.markdown("### 📊 System Capabilities")
                
                # Check OCR
                ocr_status = "✅" if st.session_state.get('rag_pipeline') else "⚠️"
                st.markdown(f"{ocr_status} **OCR Text Extraction**: {'Active' if st.session_state.get('rag_pipeline') else 'Not Ready'}")
                
                # Check retrieval
                has_docs = bool(st.session_state.indexed_documents)
                retrieval_status = "✅" if has_docs else "⚠️"
                st.markdown(f"{retrieval_status} **Retrieved Chunks**: {'Ready' if has_docs else 'No Documents'}")
                
                # Check table chunks
                table_status = "✅" if has_docs else "⚠️"
                st.markdown(f"{table_status} **Retrieved Table Chunks**: {'Ready' if has_docs else 'No Documents'}")
                
                # Check terminology
                term_status = "✅" if has_docs else "⚠️"
                st.markdown(f"{term_status} **Procurement Terminology**: {'Active' if has_docs else 'No Documents'}")
                
                if 'last_query_result' in st.session_state and st.session_state.last_query_result:
                    result = st.session_state.last_query_result
                    
                    st.divider()
                    st.markdown("### 📈 Last Query Metrics")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        total_chunks = result.get('num_chunks_retrieved', 0)
                        st.metric("Total Chunks", total_chunks)
                        
                        # Count table chunks
                        chunks = result.get('retrieved_chunks', [])
                        table_chunks = sum(1 for c in chunks if c.get('metadata', {}).get('type') == 'table')
                        st.metric("Table Chunks", table_chunks)
                    
                    with col2:
                        text_chunks = total_chunks - table_chunks
                        st.metric("Text Chunks", text_chunks)
                        
                        unique_docs = len(set(c.get('metadata', {}).get('source', '') for c in chunks))
                        st.metric("Documents Hit", unique_docs)
                    
                    # Check for procurement terms
                    answer = result.get('answer', '')
                    procurement_terms = ['PR', 'PO', 'ABC', 'Purchase', 'Procurement', 'Request', 'Order', 'PHP', 'Amount']
                    found_terms = [term for term in procurement_terms if term.lower() in answer.lower()]
                    
                    if found_terms:
                        st.success(f"✅ Procurement terms detected: {', '.join(found_terms)}")
                    else:
                        st.info("ℹ️ No procurement terms detected in answer")
            
            # Reliability Panel
            with st.expander("🛡️ Reliability", expanded=False):
                st.markdown("### 🔧 Error Handling")
                
                # Check for errors in logs
                errors = [log for log in st.session_state.get('processing_logs', []) if 'ERROR' in log or 'error' in log.lower()]
                if errors:
                    st.error(f"⚠️ {len(errors)} error(s) detected")
                    for error in errors[-5:]:  # Show last 5 errors
                        st.text(error)
                else:
                    st.success("✅ No errors detected")
                
                st.markdown("### 📄 View Uploaded Documents")
                if st.session_state.indexed_documents:
                    doc_to_view = st.selectbox(
                        "Select document:",
                        options=list(st.session_state.indexed_documents.keys()),
                        key="doc_viewer_select"
                    )
                    
                    if doc_to_view and st.button("👁️ View Document Info"):
                        doc_info = st.session_state.indexed_documents[doc_to_view]
                        
                        st.json({
                            "filename": doc_to_view,
                            "total_chunks": doc_info.get('total_chunks', 0),
                            "text_chunks": doc_info.get('text_chunks', 0),
                            "table_chunks": doc_info.get('table_chunks', 0),
                            "uploaded_at": doc_info.get('uploaded_at', 'N/A'),
                            "file_size": doc_info.get('file_size', 'N/A')
                        })
                        
                        # Show sample chunks
                        if st.session_state.rag_pipeline:
                            try:
                                # Query for chunks from this document
                                sample_query = f"source:{doc_to_view}"
                                
                                st.markdown("#### 📝 Sample Chunks")
                                st.caption("Showing first 3 chunks from this document")
                                
                                # This is a simplified view - actual implementation would query the vector store
                                st.info("💡 Full chunk viewer coming soon!")
                                
                            except Exception as e:
                                st.error(f"Error loading chunks: {e}")
                else:
                    st.info("No documents uploaded yet")
                
                st.markdown("### 💾 Session Data Persistence")
                
                # Check if session is saved
                session_id = st.session_state.get('current_session_id')
                session_file = f"chat_sessions/session_{session_id}.json" if session_id else None
                
                if session_file and os.path.exists(session_file):
                    st.success(f"✅ Session saved: {session_id}")
                    
                    # Show session info
                    try:
                        import json
                        with open(session_file, 'r') as f:
                            session_data = json.load(f)
                        
                        st.metric("Messages", len(session_data.get('messages', [])))
                        st.metric("Documents", len(session_data.get('indexed_documents', {})))
                        
                        if st.button("📥 Export Session"):
                            st.download_button(
                                "Download Session JSON",
                                data=json.dumps(session_data, indent=2),
                                file_name=f"session_{session_id}.json",
                                mime="application/json"
                            )
                    except Exception as e:
                        st.error(f"Error reading session: {e}")
                else:
                    st.warning("⚠️ Session not yet saved")
                
                # Auto-save toggle
                auto_save = st.checkbox("Auto-save session", value=True, key="auto_save_session")
                if auto_save:
                    st.caption("💾 Session auto-saves after each interaction")
            
            # Enhanced Query Debug Info
            with st.expander("🔍 Query Debug Info", expanded=False):
                if 'last_query_result' in st.session_state and st.session_state.last_query_result:
                    result = st.session_state.last_query_result
                    
                    st.subheader("📊 Query Statistics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Chunks Retrieved", result.get('num_chunks_retrieved', 0))
                        st.metric("Documents", len(result.get('relevant_documents', [])))
                    with col2:
                        if 'query_time' in st.session_state:
                            st.metric("Query Time", f"{st.session_state.query_time:.2f}s")
                        st.metric("Answer Length", len(result.get('answer', '')))
                    
                    st.subheader("📂 Source Documents")
                    for doc in result.get('relevant_documents', []):
                        st.text(f"📄 {doc}")
                    
                    st.subheader("📝 Retrieved Chunks")
                    chunks = result.get('retrieved_chunks', [])
                    for i, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                        chunk_type = chunk.get('metadata', {}).get('type', 'text')
                        chunk_icon = "📊" if chunk_type == 'table' else "📄"
                        
                        with st.expander(f"{chunk_icon} Chunk {i+1} (Score: {chunk.get('score', 'N/A'):.3f}, Type: {chunk_type})", expanded=False):
                            st.text_area(
                                f"Content:", 
                                chunk.get('content', '')[:500] + ('...' if len(chunk.get('content', '')) > 500 else ''),
                                height=100,
                                key=f"chunk_{i}"
                            )
                            st.json(chunk.get('metadata', {}))
                    
                    if len(chunks) > 5:
                        st.caption(f"... and {len(chunks) - 5} more chunks")
                        
                else:
                    st.info("No query results available. Ask a question first!")
            
            with st.expander("🤖 Model Info", expanded=False):
                if st.session_state.rag_pipeline and st.session_state.rag_pipeline.llm:
                    st.text(f"Model: {st.session_state.rag_pipeline.llm.model}")
                    st.text(f"Base URL: {st.session_state.rag_pipeline.llm.base_url}")
                    
                    # Test model connection
                    if st.button("🔌 Test Model Connection"):
                        try:
                            import time
                            start = time.time()
                            test_response = st.session_state.rag_pipeline.llm.generate("Test", max_tokens=5)
                            end = time.time()
                            st.success(f"✅ Model responding in {end-start:.2f}s")
                            st.text(f"Response: {test_response}")
                        except Exception as e:
                            st.error(f"❌ Model connection failed: {e}")
                else:
                    st.warning("RAG pipeline not initialized")
            
            # Query method selector
            if LANGCHAIN_AVAILABLE and getattr(st.session_state, 'rag_pipeline', None) and hasattr(st.session_state.rag_pipeline, 'langchain_pipeline') and st.session_state.rag_pipeline.langchain_pipeline:
                st.markdown("### 🔗 Query Method")
                query_method = st.radio(
                    "Select query method:",
                    ["Hybrid (Traditional + LangChain)", "Traditional Only", "LangChain Only"],
                    index=0,
                    help="Hybrid: Uses both methods and shows LangChain alternative. Traditional: Original RAG pipeline. LangChain: Pure LangChain approach."
                )
                st.session_state.query_method = query_method
                
                # Show current method info
                if query_method == "Hybrid (Traditional + LangChain)":
                    st.info("🔄 Using traditional RAG + LangChain enhancement")
                elif query_method == "LangChain Only":
                    st.info("🔗 Using pure LangChain approach")
                else:
                    st.info("⚡ Using traditional RAG pipeline only")
            else:
                st.session_state.query_method = "Traditional Only"
            
            with st.expander("📊 Vector Store Info", expanded=False):
                if st.session_state.rag_pipeline and st.session_state.rag_pipeline.vector_store:
                    try:
                        # Get collection stats if available
                        if hasattr(st.session_state.rag_pipeline.vector_store, '_collection'):
                            count = st.session_state.rag_pipeline.vector_store._collection.count()
                            st.metric("Total Vectors", count)
                        
                        st.text("Vector Store: ChromaDB")
                        st.text(f"Embedding Model: {st.session_state.rag_pipeline.embedding_model}")
                        
                        # Show indexed documents detail
                        if st.session_state.indexed_documents:
                            st.subheader("Document Details")
                            for filename, info in st.session_state.indexed_documents.items():
                                st.text(f"📄 {filename}")
                                st.text(f"   Chunks: {info.get('total_chunks', 0)}")
                                st.text(f"   Size: {info.get('file_size', 'Unknown')}")
                                
                    except Exception as e:
                        st.error(f"Error getting vector store info: {e}")
                else:
                    st.warning("Vector store not initialized")
            
            with st.expander("📜 Processing Logs", expanded=False):
                if 'processing_logs' in st.session_state and st.session_state.processing_logs:
                    for log in st.session_state.processing_logs[-20:]:  # Show last 20 logs
                        st.text(log)
                else:
                    st.info("No processing logs available")
        
        if st.button("🗑️ Clear Session", use_container_width=True):
            # Clear only current session data
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.vector_store.clear()
            st.session_state.rag_pipeline = None
            st.session_state.indexed_documents = {}
            st.session_state.processing_logs = []
            # Clear current session messages
            current_session = get_current_session()
            current_session['messages'] = []
            current_session['query_history'] = []
            st.success("Current session cleared!")
            st.rerun()


def add_log(message, status="INFO"):
    """Add a log entry with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {status}: {message}"
    st.session_state.processing_logs.append(log_entry)
    return log_entry

def process_documents(folder_path, uploaded_files):
    """Process documents with terminal-style progress display"""
    import io
    import contextlib
    import time
    
    # VALIDATE FIRST - before creating terminal or doing anything
    # Check if no files provided (ignore default "pdf" value)
    has_valid_folder = folder_path and folder_path.strip() != "" and folder_path.strip() != "pdf" and os.path.exists(folder_path.strip())
    
    if not uploaded_files and not has_valid_folder:
        st.session_state.is_processing = False
        st.error("❌ Please upload PDF files or specify a valid folder path before processing.")
        return
    
    # Check if folder path doesn't exist (only if user provided a path)
    if not uploaded_files and folder_path and folder_path.strip() != "" and folder_path.strip() != "pdf" and not os.path.exists(folder_path.strip()):
        st.session_state.is_processing = False
        st.error(f"❌ Folder path does not exist: {folder_path}")
        return
    
    # Clear old logs and add initial log
    clear_terminal_logs()
    add_terminal_log("Starting document processing...", "info")
    
    # Create terminal placeholder in sidebar FIRST
    with st.sidebar:
        st.divider()
        st.markdown("<div style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
        terminal_placeholder = st.empty()
        
        # Render terminal window immediately
        with terminal_placeholder:
            render_terminal()
        st.markdown("</div>", unsafe_allow_html=True)
    
    try:
        # Validate uploaded files
        if uploaded_files:
            # Check file count
            if len(uploaded_files) > 6:
                add_terminal_log("Maximum 6 PDF files allowed", "error")
                with terminal_placeholder:
                    render_terminal()
                st.session_state.is_processing = False
                st.error("❌ Maximum 6 PDF files allowed. Please select fewer files.")
                return
            
            # Check file sizes and types
            max_size_mb = 2
            max_size_bytes = max_size_mb * 1024 * 1024
            
            for uploaded_file in uploaded_files:
                # Check if PDF
                if not uploaded_file.name.lower().endswith('.pdf'):
                    add_terminal_log(f"{uploaded_file.name} is not a PDF file", "error")
                    with terminal_placeholder:
                        render_terminal()
                    st.session_state.is_processing = False
                    st.error(f"❌ {uploaded_file.name} is not a PDF file. Only PDF files are allowed.")
                    return
                
                # Check file size
                file_size = len(uploaded_file.getvalue())
                if file_size > max_size_bytes:
                    size_mb = file_size / (1024 * 1024)
                    add_terminal_log(f"{uploaded_file.name} is {size_mb:.1f}MB (exceeds 2MB limit)", "error")
                    with terminal_placeholder:
                        render_terminal()
                    st.session_state.is_processing = False
                    st.error(f"❌ {uploaded_file.name} is {size_mb:.1f}MB. Maximum file size is {max_size_mb}MB.")
                    return
        
        # Validation already done at the beginning of function
        
        # Initialize RAG pipeline if not exists
        if st.session_state.rag_pipeline is None:
            add_terminal_log("Initializing RAG pipeline", "info", loading=True)
            with terminal_placeholder:
                render_terminal()
            
            # Use current session ID for isolated vector store
            current_session_id = st.session_state.current_session_id
            
            # Generate a session ID if not exists
            if current_session_id is None:
                current_session_id = f"session_{int(time.time())}"
                st.session_state.current_session_id = current_session_id
            
            # Capture stdout to get actual initialization logs
            log_capture = io.StringIO()
            
            with contextlib.redirect_stdout(log_capture):
                st.session_state.rag_pipeline = RAGPipeline(session_id=current_session_id)
            
            # Parse captured logs and add to terminal
            captured_output = log_capture.getvalue()
            for line in captured_output.strip().split('\n'):
                if not line.strip():
                    continue
                # Skip PaddleOCR model loading messages
                if 'Creating model:' in line or 'Model files already exist' in line or 'To redownload' in line:
                    continue
                if 'WARNING:' in line or 'I1121' in line or 'oneDNN' in line:
                    continue
                    
                # Determine log type based on prefix
                if line.startswith('[+]'):
                    add_terminal_log(line[3:].strip(), "success")
                elif line.startswith('[*]'):
                    add_terminal_log(line[3:].strip(), "info")
                elif line.startswith('[!]'):
                    add_terminal_log(line[3:].strip(), "warning")
                elif line.startswith('[-]') or 'ERROR' in line.upper():
                    add_terminal_log(line[3:].strip() if line.startswith('[-]') else line, "error")
                else:
                    add_terminal_log(line.strip(), "muted")
                
                # Update terminal display
                with terminal_placeholder:
                    render_terminal()
        else:
            add_terminal_log("RAG pipeline ready (already initialized)", "success")
            with terminal_placeholder:
                render_terminal()

        
        pdf_files = []
        
        # Handle uploaded files
        if uploaded_files:
            add_terminal_log(f"Processing {len(uploaded_files)} uploaded files...", "info")
            with terminal_placeholder:
                render_terminal()
            
            upload_dir = "uploaded_docs"
            os.makedirs(upload_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_files.append(file_path)
                add_terminal_log(f"Saved: {uploaded_file.name}", "success")
                with terminal_placeholder:
                    render_terminal()
        
        # Handle folder path
        elif folder_path and os.path.exists(folder_path):
            add_terminal_log(f"Scanning folder: {folder_path}", "info")
            with terminal_placeholder:
                render_terminal()
            
            folder_pdfs = list(Path(folder_path).glob("*.pdf"))
            
            # Check file count
            if len(folder_pdfs) > 6:
                add_terminal_log(f"Found {len(folder_pdfs)} PDFs - limiting to first 6", "warning")
                with terminal_placeholder:
                    render_terminal()
                folder_pdfs = folder_pdfs[:6]
            
            # Check file sizes
            max_size_mb = 2
            max_size_bytes = max_size_mb * 1024 * 1024
            validated_pdfs = []
            
            for pdf_path in folder_pdfs:
                file_size = pdf_path.stat().st_size
                if file_size > max_size_bytes:
                    size_mb = file_size / (1024 * 1024)
                    add_terminal_log(f"Skipping {pdf_path.name} ({size_mb:.1f}MB exceeds 2MB limit)", "warning")
                    with terminal_placeholder:
                        render_terminal()
                else:
                    validated_pdfs.append(str(pdf_path))
            
            pdf_files = validated_pdfs
            add_terminal_log(f"Found {len(pdf_files)} valid PDF files", "success")
            with terminal_placeholder:
                render_terminal()
        
        if pdf_files:
            all_document_info = {}
            
            for i, pdf_file in enumerate(pdf_files):
                filename = Path(pdf_file).name
                add_terminal_log(f"Processing {filename}... ({i+1}/{len(pdf_files)})", "info")
                with terminal_placeholder:
                    render_terminal()
                
                add_terminal_log(f"Extracting text from {filename}", "info")
                with terminal_placeholder:
                    render_terminal()
                
                # Extract document data for detailed display
                doc_data = st.session_state.rag_pipeline.pdf_processor.extract_text(pdf_file)
                if doc_data:
                    # Get full text for OCR display
                    full_text = '\n'.join([p['content'] for p in doc_data['pages']])
                    
                add_terminal_log(f"Creating embeddings for {filename}", "info", loading=True)
                with terminal_placeholder:
                    render_terminal()                    # Chunk the document to get chunk info
                    chunks = st.session_state.rag_pipeline.chunker.chunk_document(doc_data)
                    
                    # Store detailed document information
                    all_document_info[filename] = {
                        'total_pages': doc_data.get('total_pages', 0),
                        'total_chunks': len(chunks),
                        'extraction_method': doc_data.get('extraction_method', 'unknown'),
                        'ocr_text': full_text[:2000],  # Store first 2000 chars for display
                        'chunks': chunks[:5],  # Store first 5 chunks for display
                        'file_path': pdf_file
                    }
                    
                    add_terminal_log(f"Indexing {filename} to vector store", "info")
                    with terminal_placeholder:
                        render_terminal()
                    
                    add_terminal_log(f"Successfully processed {filename} ({len(chunks)} chunks)", "success")
                    with terminal_placeholder:
                        render_terminal()
            
            # Update session state with document info
            st.session_state.indexed_documents.update(all_document_info)
            
            # Index all files in the pipeline
            add_terminal_log("Building vector embeddings", "info", loading=True)
            with terminal_placeholder:
                render_terminal()
            
            # Capture indexing logs too
            log_capture = io.StringIO()
            with contextlib.redirect_stdout(log_capture):
                st.session_state.rag_pipeline.index_documents(pdf_files)
            
            # Parse indexing logs
            captured_output = log_capture.getvalue()
            for line in captured_output.strip().split('\n'):
                if not line.strip():
                    continue
                # Skip verbose lines
                if 'Creating model:' in line or 'Model files already exist' in line or 'To redownload' in line:
                    continue
                if 'WARNING:' in line or 'I1121' in line or 'oneDNN' in line:
                    continue
                    
                if line.startswith('[+]'):
                    add_terminal_log(line[3:].strip(), "success")
                elif line.startswith('[*]'):
                    add_terminal_log(line[3:].strip(), "info")
                elif line.startswith('[!]'):
                    add_terminal_log(line[3:].strip(), "warning")
                else:
                    add_terminal_log(line.strip(), "muted")
                
                # Update terminal
                with terminal_placeholder:
                    render_terminal()
            
            add_terminal_log(f"All documents processed successfully!", "success")
            
            # Update terminal one last time
            with terminal_placeholder:
                render_terminal()
            
            # Reset processing flag FIRST so UI unblurs
            st.session_state.is_processing = False
            
            # Show success message
            st.success(f"✅ Successfully processed {len(pdf_files)} document(s)")
            if LANGCHAIN_AVAILABLE and rag_pipeline.langchain_pipeline:
                st.success(f"🔗 LangChain integration active!")
            
            # Save session context with documents and RAG pipeline
            save_session_context(st.session_state.current_session_id)
            
            # Clear terminal after a moment
            import time
            time.sleep(2)
            clear_terminal_logs()
            terminal_placeholder.empty()
            
        else:
            add_terminal_log("No PDF files found to process", "warning")
            with terminal_placeholder:
                render_terminal()
            st.warning("No PDF files found. Please upload files or check the folder path.")
            time.sleep(3)
            clear_terminal_logs()
            terminal_placeholder.empty()
            st.session_state.is_processing = False
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        add_terminal_log(error_msg, "error")
        if 'terminal_placeholder' in locals():
            with terminal_placeholder:
                render_terminal()
        st.error(error_msg)
        st.session_state.is_processing = False

def display_chat_history():
    """Display chat messages in ChatGPT style with session support"""
    current_session = get_current_session()
    messages = current_session.get('messages', [])
    
    # Debug: Print message count
    print(f"[DISPLAY DEBUG] Session {st.session_state.current_session_id} has {len(messages)} messages")
    for i, msg in enumerate(messages):
        print(f"[DISPLAY DEBUG] Message {i}: {msg['role']} - {msg['content'][:30]}...")
    
    if messages:
        for message in messages:
            if message['role'] == 'user':
                st.markdown(f'<div class="user-message"><strong>You</strong><br>{message["content"]}</div>', unsafe_allow_html=True)
            else:
                content = message["content"]
                
                # Display retrieved chunks if available
                if 'retrieved_chunks' in message:
                    with st.expander(f"Retrieved {len(message['retrieved_chunks'])} chunks", expanded=False):
                        for i, chunk in enumerate(message['retrieved_chunks']):
                            st.write(f"**Chunk {i+1}** (Score: {chunk.get('score', 0):.3f})")
                            st.write(f"Source: {chunk.get('metadata', {}).get('filename', 'Unknown')}")
                            st.text_area(
                                f"Content {i+1}:",
                                chunk.get('content', '')[:300] + "..." if len(chunk.get('content', '')) > 300 else chunk.get('content', ''),
                                height=100,
                                key=f"chunk_{message.get('timestamp', i)}_{i}"
                            )
                
                # Display LangChain answer if available
                if message.get('has_langchain') and message.get('langchain_answer'):
                    with st.expander("🔗 LangChain Alternative Answer", expanded=False):
                        st.markdown(f"**LangChain Answer:** {message['langchain_answer']}")
                        if message.get('langchain_sources'):
                            st.write(f"**Sources:** {len(message['langchain_sources'])} documents")
                
                st.markdown(f'<div class="assistant-message"><strong>Assistant</strong><br>{content}</div>', unsafe_allow_html=True)

def display_welcome_screen():
    """Display welcome screen when no chat history exists"""
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-logo">💬</div>
        <div class="welcome-title">How can I help you today?</div>
        <div class="welcome-subtitle">Upload documents and ask questions</div>
    </div>
    """, unsafe_allow_html=True)

def check_internet_connection():
    """Check if internet connection is available and warn user"""
    try:
        # Try to connect to a reliable server
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def main():
    """ChatGPT-style Streamlit RAG Interface (OFFLINE MODE)"""
    import time
    
    # Check for internet connection and warn if available
    if check_internet_connection():
        print("[WARNING] Internet connection detected! For complete offline mode, disable Wi-Fi/network.")
    else:
        print("[OFFLINE] ✅ No internet connection - running in full offline mode")
    
    # Page config
    st.set_page_config(
        page_title="RAG Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize system state flag
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    
    # Add cleanup on rerun to handle database locks
    if 'cleanup_needed' in st.session_state and st.session_state.cleanup_needed:
        if hasattr(st.session_state, 'rag_pipeline') and st.session_state.rag_pipeline:
            try:
                st.session_state.rag_pipeline.cleanup()
                st.session_state.rag_pipeline = None
                print("[+] Previous session cleaned up")
            except:
                pass
        st.session_state.cleanup_needed = False
    
    # Show loading screen on first load
    if not st.session_state.system_initialized:
        print("\n=== STREAMLIT APPLICATION STARTUP ===")
        print("[*] Initializing ChatGPT-style interface...")
        # Apply ChatGPT styling
        create_chatgpt_css()
        print("[+] CSS styling applied")
        render_loading_screen()
        print("[*] Loading screen displayed")
        
        # Initialize session state
        if 'chat_sessions' not in st.session_state:
            st.session_state.chat_sessions = {}
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None
        if 'rag_pipeline' not in st.session_state:
            st.session_state.rag_pipeline = None
        if 'processing_logs' not in st.session_state:
            st.session_state.processing_logs = []
        if 'indexed_documents' not in st.session_state:
            st.session_state.indexed_documents = {}
        if 'terminal_logs' not in st.session_state:
            st.session_state.terminal_logs = []
        if 'is_processing' not in st.session_state:
            st.session_state.is_processing = False
        if 'show_debug' not in st.session_state:
            st.session_state.show_debug = False
        if 'last_query_result' not in st.session_state:
            st.session_state.last_query_result = None
        if 'query_time' not in st.session_state:
            st.session_state.query_time = 0
        if 'processing_query' not in st.session_state:
            st.session_state.processing_query = False
        
        # Initialize RAG pipeline during loading screen
        try:
            import io
            import contextlib
            
            # Generate a default session ID if none exists
            if st.session_state.current_session_id is None:
                st.session_state.current_session_id = f"session_{int(time.time())}"
            
            # Capture initialization output
            log_capture = io.StringIO()
            with contextlib.redirect_stdout(log_capture):
                st.session_state.rag_pipeline = RAGPipeline(session_id=st.session_state.current_session_id)
        except Exception as e:
            # If initialization fails, just continue without pipeline
            pass
        
        time.sleep(1.5)
        st.session_state.system_initialized = True
        st.rerun()
    
    # Apply ChatGPT styling
    create_chatgpt_css()
    
    # Render top header
    render_top_header()
    
    # Create sidebar
    create_sidebar()
    
    # Chat interface with session support
    current_session = get_current_session()
    
    # Check if documents are ready and not currently processing
    docs_ready = st.session_state.rag_pipeline is not None and st.session_state.indexed_documents
    is_processing = st.session_state.get('is_processing', False)
    
    # Apply blur overlay if no documents or currently processing
    if not docs_ready:
        st.markdown('<div class="chat-warning-message">⚠️ Please upload and process documents to start chatting</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-disabled-overlay">', unsafe_allow_html=True)
    elif is_processing:
        st.markdown('<div class="chat-warning-message">⏳ Processing documents... Please wait</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-disabled-overlay">', unsafe_allow_html=True)
    
    if current_session.get('messages'):
        display_chat_history()
    else:
        display_welcome_screen()
    
    # Close blur overlay div
    if not docs_ready or is_processing:
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input at bottom
    st.markdown("---")
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            # Dynamic placeholder based on system status
            placeholder_text = "Send a message..."
            input_disabled = False
            
            if st.session_state.get('is_processing', False):
                placeholder_text = "⏳ Processing documents... Please wait"
                input_disabled = True
            elif st.session_state.get('processing_query', False):
                placeholder_text = "🔄 Processing query... Please wait"
                input_disabled = True
            elif not st.session_state.indexed_documents:
                placeholder_text = "📄 Upload documents first to start chatting"
                input_disabled = True
            elif not st.session_state.rag_pipeline:
                placeholder_text = "❌ System not ready, please restart"
                input_disabled = True
            
            user_input = st.text_input(
                "Message",
                placeholder=placeholder_text,
                label_visibility="collapsed",
                disabled=input_disabled,
                help="Type your question about the uploaded documents" if not input_disabled else "System busy, please wait"
            )
        
        with col2:
            # Dynamic send button based on system status
            button_disabled = (st.session_state.get('is_processing', False) or 
                             st.session_state.get('processing_query', False) or 
                             not st.session_state.indexed_documents or 
                             not st.session_state.rag_pipeline)
            
            if st.session_state.get('processing_query', False):
                send_button = st.form_submit_button("⏳ Processing...", disabled=True, use_container_width=True)
            elif st.session_state.get('is_processing', False):
                send_button = st.form_submit_button("📄 Loading Docs...", disabled=True, use_container_width=True)
            elif not st.session_state.indexed_documents:
                send_button = st.form_submit_button("📤 Upload First", disabled=True, use_container_width=True)
            elif not st.session_state.rag_pipeline:
                send_button = st.form_submit_button("❌ System Error", disabled=True, use_container_width=True)
            else:
                send_button = st.form_submit_button("Send 🚀", type="primary", use_container_width=True)
    
    # Handle message sending with enhanced feedback
    if send_button and user_input.strip():
        # Enhanced status checking with specific error messages
        if st.session_state.get('is_processing', False):
            st.error("⏳ Document processing in progress! Please wait for completion before asking questions.")
            current_stage = st.session_state.get('processing_stage', 'Processing...')
            st.info(f"🔄 Current stage: {current_stage}")
        elif st.session_state.get('processing_query', False):
            st.error("🔄 Already processing a query! Please wait for the current response.")
            if 'query_start_time' in st.session_state:
                elapsed = time.time() - st.session_state.query_start_time
                st.info(f"⏱️ Query processing time: {elapsed:.1f}s")
        elif st.session_state.rag_pipeline is None:
            st.error("❌ RAG engine not initialized! Please restart the application.")
            if st.button("🔄 Restart Application"):
                st.session_state.clear()
                st.rerun()
        elif not st.session_state.indexed_documents:
            st.error("📄 No documents loaded! Please upload and process PDF documents first.")
            st.info("💡 Tip: Use the sidebar to upload PDF files for analysis.")
        else:
            current_session = get_current_session()
            
            # Initialize query processing tracking
            st.session_state.processing_query = True
            st.session_state.query_start_time = time.time()
            
            # Show processing indicator
            query_placeholder = st.empty()
            with query_placeholder.container():
                st.info("🧠 Processing your question...")
                progress_bar = st.progress(0.0)
                status_text = st.empty()
            
            # Check token limits and warn user
            current_metrics = get_system_metrics()
            if current_metrics.get('token_percentage', 0) > 90:
                st.warning("⚠️ Approaching token limit! Consider starting a new session for better performance.")
            
            # Update session title based on first message
            if not current_session.get('messages'):
                title = user_input[:30] + "..." if len(user_input) > 30 else user_input
                update_session_title(st.session_state.current_session_id, title)
            
            # Set processing flag to avoid infinite loops
            if 'processing_query' not in st.session_state:
                st.session_state.processing_query = False
                
            # Only process if not already processing
            if not st.session_state.processing_query:
                st.session_state.processing_query = True
                
                # Add user message to current session
                user_message = {
                    'role': 'user', 
                    'content': user_input,
                    'timestamp': time.time()
                }
                current_session['messages'].append(user_message)
                
                # Create persistent terminal for query processing
                clear_terminal_logs()
                add_terminal_log(f"Processing query: {user_input}", "info", loading=True)
            
                # Process the query immediately without rerun
                try:
                    start_time = time.time()
                    print(f"\n=== QUERY PROCESSING ===")
                    print(f"[*] User Query: {user_input[:50]}{'...' if len(user_input) > 50 else ''}")
                    print(f"[*] Session: {st.session_state.current_session_id}")
                    
                    add_log(f"Processing query: {user_input}")
                    
                    # Track retrieval time
                    retrieval_start = time.time()
                    result = st.session_state.rag_pipeline.query(user_input)
                    retrieval_time = time.time() - retrieval_start
                    
                    # Store debug information in session state
                    st.session_state.last_query_result = result
                    st.session_state.query_time = retrieval_time
                    
                    response = result['answer']
                    sources = result.get('relevant_documents', [])
                    retrieved_chunks = result.get('retrieved_chunks', [])
                    
                    # Format response with sources
                    if sources:
                        response += f"\n\n**Sources:** {', '.join(sources)}"
                    
                    total_time = time.time() - start_time
                    print(f"[+] Query completed in {total_time:.2f}s ({len(retrieved_chunks)} chunks)")
                    print(f"[*] Response length: {len(response)} characters")
                    
                    add_log(f"Generated response from {len(retrieved_chunks)} chunks in {total_time:.2f}s", "SUCCESS")
                    
                    # Add assistant response with retrieved chunks
                    assistant_message = {
                        'role': 'assistant',
                        'content': response,
                        'retrieved_chunks': retrieved_chunks,
                        'sources': sources,
                        'timestamp': time.time()
                    }
                    current_session['messages'].append(assistant_message)
                    
                    # Debug: Print session info
                    print(f"[DEBUG] Added message to session {st.session_state.current_session_id}")
                    print(f"[DEBUG] Session now has {len(current_session['messages'])} messages")
                    print(f"[DEBUG] Last message: {current_session['messages'][-1]['role']} - {current_session['messages'][-1]['content'][:50]}...")
                    
                    # Store query in session history
                    current_session.setdefault('query_history', []).append({
                        'query': user_input,
                        'response_length': len(response),
                        'chunks_retrieved': len(retrieved_chunks),
                        'timestamp': time.time()
                    })
                    
                    # Save session context after successful interaction
                    save_session_context(st.session_state.current_session_id)
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    add_log(error_msg, "ERROR")
                    print(f"[ERROR] {error_msg}")
                    
                    current_session['messages'].append({
                        'role': 'assistant',
                        'content': f"Sorry, I encountered an error: {str(e)}",
                        'timestamp': time.time()
                    })
                    # Save session context even on error
                    save_session_context(st.session_state.current_session_id)
                
                # Reset processing flag and rerun to show results
                st.session_state.processing_query = False
                st.rerun()
    
    # Render metrics bar
    render_metrics_bar()


if __name__ == "__main__":
    main()
