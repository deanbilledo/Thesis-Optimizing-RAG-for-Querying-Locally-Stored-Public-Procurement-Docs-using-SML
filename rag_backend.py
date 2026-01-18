"""
RAG Backend - Session Management, Document Processing, and Model Inference
"""

import os
import json
import torch
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import shutil
import hashlib
import time
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import warnings
warnings.filterwarnings('ignore')

# OCR support for scanned PDFs
try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    
    # Configure Tesseract path for Windows
    if os.name == 'nt':  # Windows
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
    
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("⚠️ OCR libraries not installed. Scanned PDFs will not work. Install: pip install pdf2image pytesseract pillow")

# ML-based table extraction with camelot
try:
    import camelot
    HAS_CAMELOT = True
    print("✓ Camelot table extraction available")
except ImportError:
    HAS_CAMELOT = False
    print("⚠️ Camelot not installed. Table Mode will use basic extraction. Install: pip install camelot-py[cv] opencv-python")

# Suppress transformers warnings about generation config
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Import streamlit for caching
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None


# Streamlit-cached model loaders
if HAS_STREAMLIT:
    @st.cache_resource
    def get_cached_llm_model():
        """Load and cache the LLM model globally across all sessions"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Check for local base model first (for offline use)
        local_base_model = Path(__file__).parent / 'base_model'
        
        if local_base_model.exists():
            print(f" Loading local base model from {local_base_model}")
            base_model_path = str(local_base_model)
        else:
            print(" Local base model not found, downloading from HuggingFace...")
            base_model_path = "google/gemma-2-2b-it"
        
        print(f" Loading base model on {device}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None,
            low_cpu_mem_usage=True,
        )
        
        if device == 'cpu':
            base_model = base_model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("Loading LoRA adapter...")
        adapter_path = Path(__file__).parent / "model"
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()
        
        # Keep base model for judge (without adapter)
        base_model.eval()
        
        print("✅ Model loaded successfully!")
        return {
            'model': model, 
            'base_model': base_model,
            'tokenizer': tokenizer, 
            'device': device, 
            'warmed_up': False
        }

    @st.cache_resource
    def get_cached_embedding_model():
        """Load and cache the embedding model globally across all sessions"""
        local_embedding_path = Path(__file__).parent / 'embedding_model'
        
        if local_embedding_path.exists():
            print(f"✅ Loading local embedding model from {local_embedding_path.name}")
            model_path = str(local_embedding_path)
        else:
            print("⚠️ Local embedding model not found, downloading...")
            model_path = 'sentence-transformers/all-MiniLM-L6-v2'
        
        return SentenceTransformer(model_path)


# Global model cache to share across sessions (fallback for non-Streamlit use)
_MODEL_CACHE = {
    'llm_model': None,
    'llm_tokenizer': None,
    'device': None
}

# Configuration
CONFIG = {
    'max_pdfs_per_session': 6,
    'max_pages_per_pdf': 15,
    'max_total_size_mb': 20,
    'max_sessions': 10,
    'chunk_size': 1200,
    'chunk_overlap': 50,
    'top_k_chunks': 3,  # Increased from 5 to get more document content for compliance checking
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'base_model': 'google/gemma-3-1b-it',  # MUST match your LoRA adapter
    'lora_adapter_path': './model',
    'max_new_tokens': 512,  # Increased for detailed compliance analysis
    'temperature': 0.3,  # Slightly higher for more analytical responses (was 0.1)
    'do_sample': True,  # Enable sampling for better quality (was False)
    # LLM Judge configuration
    'judge_enabled': True,  # Enable LLM-as-a-judge validation
    'judge_confidence_threshold': 60,  # Minimum confidence % to show response without warning
    'judge_max_tokens': 150,  # Judge uses fewer tokens
    'judge_temperature': 0.1,  # Judge uses lower temperature for consistency
    # Retrieval weights tuned to be cosine-dominant with minimal section tag influence
    'retrieval_weights': {
        'cosine': 0.90,
        'llm_judge': 0.00,
        'structural': 0.05,  # Minimal section tag influence
        'metadata': 0.00,
        'mmr': 0.05
    },
    # Section tags for document structure
    'section_tags': [
        'CONTRACT_AMOUNT', 'BIDDER_INFO', 'LEGAL_CLAUSES',
        'COMPLIANCE_REQUIREMENTS', 'TIMELINE', 'TECHNICAL_SPECS',
        'GENERAL', 'TABLE_DATA'
    ]
}


def check_gpu() -> Dict:
    """Check GPU availability and return info"""
    if torch.cuda.is_available():
        return {
            'available': True,
            'name': torch.cuda.get_device_name(0),
            'memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'memory_allocated': torch.cuda.memory_allocated(0) / 1e9,
        }
    return {'available': False}


def tag_section(text: str) -> str:
    """Identify document section based on keywords (lightweight tagging)"""
    text_lower = text.lower()
    
    # Simple keyword-based tagging
    if any(kw in text_lower for kw in ['amount', 'price', 'cost', 'budget', 'php', '$']):
        return 'CONTRACT_AMOUNT'
    elif any(kw in text_lower for kw in ['bidder', 'supplier', 'vendor', 'contractor']):
        return 'BIDDER_INFO'
    elif any(kw in text_lower for kw in ['legal', 'clause', 'terms', 'conditions', 'obligations']):
        return 'LEGAL_CLAUSES'
    elif any(kw in text_lower for kw in ['compliance', 'requirement', 'regulation', 'standard']):
        return 'COMPLIANCE_REQUIREMENTS'
    elif any(kw in text_lower for kw in ['timeline', 'schedule', 'deadline', 'date', 'duration']):
        return 'TIMELINE'
    elif any(kw in text_lower for kw in ['technical', 'specification', 'specs', 'requirements']):
        return 'TECHNICAL_SPECS'
    elif '|' in text or text.count('\t') > 3:  # Detect tables
        return 'TABLE_DATA'
    else:
        return 'GENERAL'


def calculate_mmr_score(query_embedding, candidate_embeddings, selected_indices, lambda_param=0.5):
    """Calculate Maximal Marginal Relevance score for diversity"""
    if not selected_indices:
        return 1.0
    
    # Simple diversity penalty based on number of already selected chunks
    diversity_penalty = 1.0 - (len(selected_indices) * 0.1)
    return max(0.1, diversity_penalty)


def generate_pdf_metadata(pdf_path: str) -> Dict:
    """Generate comprehensive metadata for PDF document"""
    metadata = {
        'filename': Path(pdf_path).name,
        'file_size_mb': Path(pdf_path).stat().st_size / (1024 * 1024),
        'total_pages': 0,
        'total_chunks': 0,
        'section_distribution': {},
        'has_tables': False,
        'keywords': [],
        'estimated_tokens': 0,
        'created_at': datetime.now().isoformat()
    }
    
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            metadata['total_pages'] = len(reader.pages)
            
            # Analyze content
            all_text = ""
            for page in reader.pages:
                all_text += page.extract_text() or ""
            
            # Extract keywords (top terms)
            words = all_text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 4 and word.isalpha():
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            metadata['keywords'] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            metadata['keywords'] = [word for word, _ in metadata['keywords']]
            metadata['estimated_tokens'] = len(words)
            metadata['has_tables'] = '|' in all_text or all_text.count('\t') > 10
            
    except Exception as e:
        metadata['error'] = str(e)
    
    return metadata


class PDFProcessor:
    """Process PDFs and extract text"""
    
    @staticmethod
    def extract_text(pdf_path: str, use_ocr_tables: bool = False, session=None) -> List[Dict]:
        """Extract text from PDF with page information - supports scanned PDFs via OCR
        
        Args:
            pdf_path: Path to PDF file
            use_ocr_tables: If True, enable Table Mode - uses LLM to structure and format tables properly.
            session: RAGSession instance (needed for LLM model access in Table Mode)
        """
        chunks = []
        
        try:
            # If Table Mode is enabled, use LLM to structure tables (NO OCR for digital PDFs)
            if use_ocr_tables and session:
                print(f"📊 Table Mode enabled: Processing with LLM for table structuring...")
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    total_pages = len(pdf_reader.pages)
                    
                    if total_pages > CONFIG['max_pages_per_pdf']:
                        raise ValueError(f"PDF has {total_pages} pages, max {CONFIG['max_pages_per_pdf']} allowed")
                
                # Use LLM processing for digital PDFs (extract text then structure with LLM)
                chunks = PDFProcessor._extract_with_llm(pdf_path, total_pages, session)
                return chunks
            
            # Otherwise, use regular text extraction (digital PDFs)
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                if total_pages > CONFIG['max_pages_per_pdf']:
                    raise ValueError(f"PDF has {total_pages} pages, max {CONFIG['max_pages_per_pdf']} allowed")
                
                # Try regular text extraction first
                has_text = False
                has_tables = False
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if text and text.strip() and len(text.strip()) > 50:
                        has_text = True
                        
                        # Split into chunks
                        chunk_texts = PDFProcessor._chunk_text(text)
                        
                        for chunk_text in chunk_texts:
                            if chunk_text.strip():  # Only add non-empty chunks
                                # Tag section for this chunk
                                section_tag = tag_section(chunk_text)
                                
                                chunks.append({
                                    'text': chunk_text,
                                    'page': page_num + 1,
                                    'source': os.path.basename(pdf_path),
                                    'section_tag': section_tag
                                })
                
                # If no text found, try OCR for scanned PDFs
                if not has_text and HAS_OCR:
                    print(f"📄 Scanned PDF detected: {os.path.basename(pdf_path)} - Using OCR...")
                    chunks = PDFProcessor._extract_with_ocr(pdf_path, total_pages, session=None)
                elif not has_text and not HAS_OCR:
                    raise Exception(
                        "This appears to be a scanned PDF (no extractable text). "
                        "Please install OCR support: pip install pdf2image pytesseract pillow"
                    )
        
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
        
        return chunks
    
    @staticmethod
    def _extract_with_llm(pdf_path: str, total_pages: int, session) -> List[Dict]:
        """Extract text from digital PDF and use LLM to structure tables
        Creates ONE CHUNK PER PAGE with LLM-structured content"""
        chunks = []
        
        # Load LLM model ONCE at start
        if session._llm_model is None:
            print("  📥 Loading LLM model for table formatting...")
            session.load_llm()
            print("  ✓ Model loaded")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                print(f"  Processing {total_pages} pages in Table Mode (structure-preserving extraction)...")
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    
                    # Extract text with proper ordering - group by rows, sort by columns
                    text_elements = []
                    
                    def visitor_body(text, cm, tm, font_dict, font_size):
                        x = tm[4]  # X coordinate (horizontal position)
                        y = tm[5]  # Y coordinate (vertical position)
                        if text.strip():
                            text_elements.append((y, x, text))
                    
                    # Extract with visitor to get positions
                    try:
                        page.extract_text(visitor_text=visitor_body)
                        
                        # Group elements by row (similar Y positions = same row)
                        rows = {}
                        for y, x, text in text_elements:
                            # Round Y to group elements on same row (tolerance of 5 units for better grouping)
                            row_key = round(y / 5) * 5
                            if row_key not in rows:
                                rows[row_key] = []
                            rows[row_key].append((x, text))
                        
                        # Sort rows top to bottom, sort elements in each row left to right
                        sorted_rows = sorted(rows.items(), key=lambda item: -item[0])
                        
                        # Build text with proper structure
                        lines = []
                        for row_y, elements in sorted_rows:
                            # Sort elements in row by X position (left to right)
                            elements.sort(key=lambda e: e[0])
                            # Join elements in row with space to separate fields
                            row_text = ' '.join([e[1].strip() for e in elements])
                            if row_text.strip():
                                lines.append(row_text.strip())
                        
                        text = '\n'.join(lines)
                    except:
                        # Fallback to regular extraction
                        text = page.extract_text()
                    
                    # Try to extract form field values (for fillable PDFs)
                    form_data = []
                    try:
                        if '/AcroForm' in pdf_reader.trailer['/Root']:
                            fields = pdf_reader.get_form_text_fields()
                            if fields:
                                for field_name, field_value in fields.items():
                                    if field_value:
                                        form_data.append(f"{field_name}: {field_value}")
                    except:
                        pass
                    
                    # Combine form data with extracted text
                    if form_data:
                        text = '\n'.join(form_data) + '\n\n' + text
                    
                    if text and text.strip():
                        # Apply smart cleaning (remove duplicates, format tables)
                        text = PDFProcessor._smart_clean_tables(text)
                        
                        # Create ONE CHUNK PER PAGE
                        if text:
                            section_tag = tag_section(text)
                            
                            chunks.append({
                                'text': text,
                                'page': page_num + 1,
                                'source': os.path.basename(pdf_path),
                                'section_tag': section_tag
                            })
                        
                        print(f"  ✓ Page {page_num + 1}/{total_pages}")
        
        except Exception as e:
            raise Exception(f"LLM processing failed: {str(e)}")
        
        return chunks
    
    @staticmethod
    def _extract_with_ocr(pdf_path: str, total_pages: int, session=None) -> List[Dict]:
        """Extract text from scanned PDF using OCR
        Creates ONE CHUNK PER PAGE and uses LLM to clean/structure text and extract tables"""
        chunks = []
        
        # Load LLM model if session provided and not already loaded
        if session and session._llm_model is None:
            print("  Loading LLM model for table extraction...")
            session.load_llm()
        
        try:
            # Set poppler path for Windows
            poppler_path = None
            if os.name == 'nt':
                poppler_locations = [
                    os.path.join(os.environ.get('LOCALAPPDATA', ''), 'poppler', 'Library', 'bin'),
                    r'C:\Program Files\poppler\Library\bin',
                    r'C:\poppler\Library\bin'
                ]
                for loc in poppler_locations:
                    if os.path.exists(loc):
                        poppler_path = loc
                        break
            
            # Convert PDF pages to images
            if poppler_path:
                images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)
            else:
                images = convert_from_path(pdf_path, dpi=300)
            
            for page_num, image in enumerate(images):
                if page_num >= total_pages:
                    break
                
                # Perform OCR on the full image (raw extraction)
                text = pytesseract.image_to_string(image, lang='eng')
                
                if text and text.strip():
                    # Basic cleanup only
                    text = PDFProcessor._clean_ocr_text(text)
                    
                    # Use LLM to clean, structure, and extract tables from OCR text
                    # LLM will fix formatting, align tables, and make text readable
                    cleaned_text = PDFProcessor._llm_clean_ocr_text(text, session)
                    
                    # Create ONE CHUNK PER PAGE (no splitting)
                    if cleaned_text.strip():
                        # Tag section for this page
                        section_tag = tag_section(cleaned_text)
                        
                        chunks.append({
                            'text': cleaned_text,
                            'page': page_num + 1,
                            'source': os.path.basename(pdf_path),
                            'section_tag': section_tag
                        })
                
                print(f"  ✓ Processed page {page_num + 1}/{len(images)} via OCR + LLM cleanup")
        
        except Exception as e:
            raise Exception(f"OCR processing failed: {str(e)}")
        
        return chunks
    
    @staticmethod
    def _extract_tables_with_camelot(pdf_path: str, page_num: int) -> str:
        """Extract tables using Camelot ML-based detection and clean the output"""
        try:
            if not HAS_CAMELOT:
                return None
            
            # Extract tables from specific page
            tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_num + 1),
                flavor='stream',  # Stream is better for complex forms
                suppress_stdout=True,
                edge_tol=50
            )
            
            # If no tables found, try lattice
            if len(tables) == 0:
                tables = camelot.read_pdf(
                    pdf_path,
                    pages=str(page_num + 1),
                    flavor='lattice',
                    suppress_stdout=True
                )
            
            if len(tables) == 0:
                return None
            
            # Convert tables to clean format
            result = []
            for idx, table in enumerate(tables):
                if table.accuracy > 40:  # Lower threshold for forms
                    df = table.df
                    
                    # Remove completely empty columns
                    df = df.loc[:, (df != '').any(axis=0)]
                    
                    # Remove rows where all cells are empty
                    df = df.loc[(df != '').any(axis=1)]
                    
                    if df.empty:
                        continue
                    
                    # Convert to simple text format (easier to read than markdown table)
                    table_lines = []
                    table_lines.append(f"\n=== TABLE {idx+1} ===")
                    
                    for _, row in df.iterrows():
                        # Join non-empty cells with "|"
                        cells = [str(cell).strip() for cell in row if str(cell).strip()]
                        if cells:
                            table_lines.append(' | '.join(cells))
                    
                    table_lines.append("=" * 40)
                    result.append('\n'.join(table_lines))
            
            return '\n'.join(result) if result else None
            
        except Exception as e:
            print(f"    Camelot extraction failed: {str(e)}")
            return None
    
    @staticmethod
    def _smart_clean_tables(text: str) -> str:
        """Smart cleaning for table content - preserves structure, removes duplicates"""
        import re
        
        lines = text.split('\n')
        
        # Clean each line but preserve structure
        cleaned_lines = []
        prev_was_empty_field = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is an empty field label (ends with : but nothing after)
            is_empty_field = line.endswith(':') and len(line) < 50
            
            # If we have consecutive empty field labels, merge them on same line
            if is_empty_field and prev_was_empty_field and cleaned_lines:
                # Don't add blank line, keep compact
                pass
            
            # Check if line has table-like structure (multiple spaces/tabs)
            if '  ' in line or '\t' in line:
                # Split by multiple spaces or tabs
                parts = re.split(r'\s{2,}|\t+', line)
                # Join with pipe for readability
                line = ' | '.join(p.strip() for p in parts if p.strip())
            
            cleaned_lines.append(line)
            prev_was_empty_field = is_empty_field
        
        # Remove consecutive duplicate lines (keep first occurrence)
        final_lines = []
        prev_line = None
        for line in cleaned_lines:
            if line != prev_line:  # Only check consecutive duplicates
                final_lines.append(line)
                prev_line = line
        
        return '\n'.join(final_lines)
    
    @staticmethod
    def _llm_format_page(text: str, session) -> str:
        """Use LLM to format page text - fast and controlled"""
        try:
            llm_model = session._llm_model
            llm_tokenizer = session._llm_tokenizer
            device = session._device if hasattr(session, '_device') else 'cpu'
            
            if llm_model is None or llm_tokenizer is None:
                return text
            
            # Truncate input to prevent long processing
            text = text[:1500]
            
            # Simple prompt
            prompt = f"""Reformat this document page. Organize the information clearly. Keep all details.

{text}

Reformatted:"""
            
            # Tokenize with strict limits
            inputs = llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(device)
            
            # Generate with strict limits to prevent infinite generation
            with torch.no_grad():
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=512,  # Strict limit
                    min_new_tokens=50,   # Ensure some output
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=llm_tokenizer.eos_token_id,
                    eos_token_id=llm_tokenizer.eos_token_id,
                    num_beams=1,
                    early_stopping=True  # Stop as soon as possible
                )
            
            # Decode
            result = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract formatted part
            if "Reformatted:" in result:
                result = result.split("Reformatted:")[-1].strip()
            
            return result if result else text
            
        except Exception as e:
            print(f"    LLM format error: {str(e)}")
            return text
    
    @staticmethod
    def _llm_clean_ocr_text(text: str, session=None) -> str:
        """Use LLM to clean OCR text, extract and structure tables, fix formatting"""
        try:
            # Get model from session if available
            if session:
                llm_model = session._llm_model
                llm_tokenizer = session._llm_tokenizer
                device = session._device if hasattr(session, '_device') else 'cpu'
            else:
                # Fallback to global cache
                global _MODEL_CACHE
                llm_model = _MODEL_CACHE.get('llm_model')
                llm_tokenizer = _MODEL_CACHE.get('llm_tokenizer')
                device = _MODEL_CACHE.get('device', 'cpu')
            
            # If model not loaded, skip LLM cleanup (fallback to basic cleaning)
            if llm_model is None or llm_tokenizer is None:
                print("  ⚠️ LLM not available, using basic text cleaning")
                return text
            
            # Prepare prompt for LLM to clean OCR text and format tables
            prompt = f"""Clean up and organize this document text. Format any tables you find using markdown table syntax with pipes (|). Keep all numbers, names, and details exactly as shown.

Document text:
{text[:2000]}

Formatted document:"""
            
            # Tokenize
            inputs = llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500
            ).to(device)
            
            # Generate cleaned text (fast settings)
            with torch.no_grad():
                outputs = llm_model.generate(
                    **inputs,
                    max_new_tokens=800,  # Reduced for speed
                    temperature=0.1,
                    do_sample=False,  # Greedy decoding is faster
                    pad_token_id=llm_tokenizer.eos_token_id,
                    num_beams=1  # No beam search for speed
                )
            
            # Decode output
            cleaned = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the formatted text
            if "Formatted document:" in cleaned:
                cleaned = cleaned.split("Formatted document:")[-1].strip()
            
            return cleaned if cleaned else text
            
        except Exception as e:
            print(f"  ⚠️ LLM cleanup failed: {e}, using basic cleaning")
            return text
    
    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        """Basic OCR text cleaning (fallback when LLM not available)"""
        import re
        
        # Remove excessive whitespace but preserve line breaks
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            line = line.strip()
            
            # Skip very short lines with only symbols (OCR noise)
            if len(line) < 3 and not line.isalnum():
                continue
            
            # Fix common OCR errors
            line = re.sub(r'\s+', ' ', line)  # Multiple spaces to single
            line = line.replace('|', '')  # Remove vertical bars (scan artifacts)
            
            # Preserve the line
            if line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def _chunk_text(text: str) -> List[str]:
        """Simple sequential chunking - processes text from top to bottom with fixed size and overlap"""
        chunk_size = CONFIG['chunk_size']
        overlap = CONFIG['chunk_overlap']
        
        # Split into lines for better readability
        lines = text.split('\n')
        
        chunks = []
        current_chunk_lines = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            
            # If adding this line would exceed chunk size and we have content, finalize chunk
            if current_size + line_size > chunk_size and current_chunk_lines:
                # Create chunk from accumulated lines
                chunk_text = '\n'.join(current_chunk_lines).strip()
                if chunk_text:  # Only add non-empty chunks
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                # Keep last few lines for context (overlap)
                overlap_lines = []
                overlap_size = 0
                for prev_line in reversed(current_chunk_lines):
                    line_len = len(prev_line) + 1
                    if overlap_size + line_len <= overlap:
                        overlap_lines.insert(0, prev_line)
                        overlap_size += line_len
                    else:
                        break
                
                current_chunk_lines = overlap_lines
                current_size = overlap_size
            
            # Add current line to chunk
            current_chunk_lines.append(line)
            current_size += line_size
        
        # Add final chunk if any content remains
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines).strip()
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks


class RAGSession:
    """Individual RAG session with isolated context"""
    
    def __init__(self, session_id: str, session_name: str, session_dir: Path):
        self.session_id = session_id
        self.session_name = session_name
        self.session_dir = session_dir
        self.pdf_dir = session_dir / "pdfs"
        
        # Initialize permanent knowledge base client
        self._kb_client = None
        self._kb_collection = None
        self.db_dir = session_dir / "chroma_db"
        self.metadata_file = session_dir / "metadata.json"
        
        # Create directories
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Initialize models (lazy loading)
        self._embedding_model = None
        self._llm_model = None
        self._base_model = None  # For LLM judge (no adapter)
        self._llm_tokenizer = None
        self._chroma_client = None
        self._collection = None
        
        # Chat history
        self.chat_history = self.metadata.get('chat_history', [])
        
        # Documents
        self.documents = self.metadata.get('documents', [])
    
    def __del__(self):
        """Cleanup method to properly close ChromaDB connections"""
        self.close()
    
    def close(self):
        """Explicitly close ChromaDB connections to release file locks"""
        try:
            if self._collection is not None:
                del self._collection
                self._collection = None
            
            if self._chroma_client is not None:
                del self._chroma_client
                self._chroma_client = None
            
            if self._kb_collection is not None:
                del self._kb_collection
                self._kb_collection = None
            
            if self._kb_client is not None:
                del self._kb_client
                self._kb_client = None
                
            # Small delay to ensure cleanup
            import time
            time.sleep(0.1)
        except Exception as e:
            # Silently handle cleanup errors
            pass
    
    def _get_knowledge_base(self):
        """Lazy load permanent knowledge base collection"""
        if self._kb_collection is None:
            try:
                kb_dir = Path('./knowledge_base/chroma_db')
                if kb_dir.exists():
                    self._kb_client = chromadb.PersistentClient(
                        path=str(kb_dir),
                        settings=Settings(anonymized_telemetry=False)
                    )
                    self._kb_collection = self._kb_client.get_collection("permanent_knowledge_base")
                    print("✅ Connected to permanent knowledge base")
                else:
                    print("⚠️ Knowledge base not found. Run knowledge_base_setup.py first.")
            except Exception as e:
                print(f"⚠️ Could not load knowledge base: {e}")
        return self._kb_collection
    
    @property
    def _config(self):
        """Expose config for UI access"""
        return CONFIG
    
    def _load_metadata(self) -> Dict:
        """Load session metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    # Ensure documents is a list
                    if 'documents' in metadata and not isinstance(metadata['documents'], list):
                        metadata['documents'] = []
                    # Ensure chat_history is a list
                    if 'chat_history' in metadata and not isinstance(metadata['chat_history'], list):
                        metadata['chat_history'] = []
                    return metadata
            except (json.JSONDecodeError, Exception) as e:
                print(f"Warning: Could not load metadata, creating new: {e}")
                # Return default metadata if loading fails
        
        return {
            'session_id': self.session_id,
            'session_name': self.session_name,
            'created_at': datetime.now().isoformat(),
            'documents': [],
            'chat_history': []
        }
    
    def _save_metadata(self):
        """Save session metadata"""
        # Ensure documents is a list of dicts (JSON serializable)
        self.metadata['documents'] = self.documents if isinstance(self.documents, list) else []
        self.metadata['chat_history'] = self.chat_history if isinstance(self.chat_history, list) else []
        self.metadata['updated_at'] = datetime.now().isoformat()
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    @property
    def embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None:
            # Try Streamlit cache first
            if HAS_STREAMLIT and st is not None:
                try:
                    self._embedding_model = get_cached_embedding_model()
                    print("✅ Using Streamlit cached embedding model")
                    return self._embedding_model
                except Exception as e:
                    print(f"⚠️ Streamlit cache failed: {e}, loading normally")
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Try local model first (for offline use)
            local_model_path = Path('./embedding_model')
            if local_model_path.exists():
                print(f"✅ Loading local embedding model from {local_model_path}")
                self._embedding_model = SentenceTransformer(
                    str(local_model_path),
                    device=device
                )
            else:
                # Fallback to downloading from HuggingFace (requires internet)
                print(f"⏳ Downloading embedding model from HuggingFace...")
                self._embedding_model = SentenceTransformer(
                    CONFIG['embedding_model'],
                    device=device
                )
                # Save locally for future offline use
                print(f"💾 Saving embedding model locally for offline use...")
                local_model_path.mkdir(parents=True, exist_ok=True)
                self._embedding_model.save(str(local_model_path))
                print(f"✅ Embedding model saved to {local_model_path}")
        return self._embedding_model
    
    @property
    def chroma_client(self):
        """Lazy load ChromaDB with proper connection settings"""
        if self._chroma_client is None:
            try:
                # Configure ChromaDB settings to prevent file locking issues
                chroma_settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
                
                # Add retry logic for database lock errors
                max_retries = 5
                retry_delay = 0.5
                
                for attempt in range(max_retries):
                    try:
                        self._chroma_client = chromadb.PersistentClient(
                            path=str(self.db_dir),
                            settings=chroma_settings
                        )
                        self._collection = self._chroma_client.get_or_create_collection(
                            name=f"session_{self.session_id}"
                        )
                        break  # Success
                    except Exception as retry_error:
                        if "being used by another process" in str(retry_error) and attempt < max_retries - 1:
                            print(f"⏳ Database locked, retrying ({attempt + 1}/{max_retries})...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                        else:
                            raise  # Re-raise if it's the last attempt or different error
            except Exception as e:
                # If there's a schema error or file lock error, delete and recreate the database
                if "no such column" in str(e).lower() or "being used by another process" in str(e):
                    print(f"⚠️ Database issue detected. Resetting database...")
                    
                    # Force close any existing connections
                    if self._chroma_client is not None:
                        try:
                            del self._collection
                            del self._chroma_client
                        except:
                            pass
                        self._chroma_client = None
                        self._collection = None
                    
                    # Wait for connections to close
                    time.sleep(2)
                    
                    # Force delete with retries
                    import shutil
                    max_delete_retries = 5
                    for del_attempt in range(max_delete_retries):
                        try:
                            if self.db_dir.exists():
                                shutil.rmtree(self.db_dir)
                            break
                        except Exception as del_err:
                            if del_attempt < max_delete_retries - 1:
                                print(f"⏳ Waiting for database to unlock ({del_attempt + 1}/{max_delete_retries})...")
                                time.sleep(1)
                            else:
                                print(f"⚠️ Could not delete database: {del_err}")
                                print("💡 Please close Streamlit and restart the app to reset the database.")
                                raise
                    
                    self.db_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Reinitialize with retry logic
                    time.sleep(1)
                    
                    chroma_settings = Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                    
                    self._chroma_client = chromadb.PersistentClient(
                        path=str(self.db_dir),
                        settings=chroma_settings
                    )
                    self._collection = self._chroma_client.get_or_create_collection(
                        name=f"session_{self.session_id}"
                    )
                    print("✅ Database reset successfully. Please re-upload your documents.")
                else:
                    raise
        return self._chroma_client
    
    @property
    def collection(self):
        """Get ChromaDB collection"""
        if self._collection is None:
            _ = self.chroma_client  # Initialize client
        return self._collection
    
    def load_llm(self):
        """Load LLM with LoRA adapter (using Streamlit cache or global cache)"""
        # Try Streamlit cache first
        if HAS_STREAMLIT and st is not None:
            try:
                cached_data = get_cached_llm_model()
                self._llm_model = cached_data['model']
                self._base_model = cached_data.get('base_model')  # For judge
                self._llm_tokenizer = cached_data['tokenizer']
                self._device = cached_data['device']
                print("✅ Using Streamlit cached model")
                return
            except Exception as e:
                print(f"⚠️ Streamlit cache failed: {e}, falling back to global cache")
        
        # Fallback to global cache
        global _MODEL_CACHE
        
        # Check if model is already loaded in cache
        if _MODEL_CACHE['llm_model'] is not None:
            self._llm_model = _MODEL_CACHE['llm_model']
            self._llm_tokenizer = _MODEL_CACHE['llm_tokenizer']
            self._device = _MODEL_CACHE['device']
            print("✅ Using cached model")
            return
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Check for local base model first (for offline use)
        local_base_model = Path(__file__).parent / 'base_model'
        if local_base_model.exists():
            base_model_path = str(local_base_model)
            print(f"✅ Loading local base model from {base_model_path}")
        else:
            base_model_path = CONFIG['base_model']
            print(f"⏳ Downloading base model from HuggingFace: {base_model_path}")
        
        print(f"⏳ Loading base model on {device}...")
            
        # Load tokenizer
        self._llm_tokenizer = AutoTokenizer.from_pretrained(
            CONFIG['lora_adapter_path']
        )
        
        # Load base model with proper device placement
        if device == 'cuda':
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                dtype=torch.float16,
                device_map='auto',
                low_cpu_mem_usage=True
            )
        else:
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        
        # Load LoRA adapter
        print("Loading LoRA adapter...")
        self._llm_model = PeftModel.from_pretrained(
            self._llm_model,
            CONFIG['lora_adapter_path']
        )
        
        # Ensure model is on correct device
        if device == 'cpu':
            self._llm_model = self._llm_model.to(device)
        
        self._llm_model.eval()
        
        # Store device for later use
        self._device = device
        
        # Cache the model globally
        _MODEL_CACHE['llm_model'] = self._llm_model
        _MODEL_CACHE['llm_tokenizer'] = self._llm_tokenizer
        _MODEL_CACHE['device'] = self._device
        
        print("✅ Model loaded successfully!")
    
    def warmup_model(self):
        """Warm up the model with a simple query to avoid first-query delay"""
        # Load model if not already loaded
        if self._llm_model is None:
            self.load_llm()
        
        # Check Streamlit cache warmup status
        if HAS_STREAMLIT and st is not None:
            try:
                cached_data = get_cached_llm_model()
                if cached_data.get('warmed_up', False):
                    print("✅ Model already warmed up (Streamlit cache)")
                    return
            except:
                pass
        
        # Skip warmup if model was already cached (already warmed up)
        global _MODEL_CACHE
        if _MODEL_CACHE.get('warmed_up', False):
            return
        
        # Generate a simple response to warm up CUDA kernels
        warmup_text = "Hello"
        inputs = self._llm_tokenizer(
            warmup_text,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self._device)
        
        # Quick generation to warm up
        with torch.no_grad():
            _ = self._llm_model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False
            )
        
        # Mark as warmed up in both caches
        _MODEL_CACHE['warmed_up'] = True
        if HAS_STREAMLIT and st is not None:
            try:
                cached_data = get_cached_llm_model()
                cached_data['warmed_up'] = True
            except:
                pass
        
        print("✅ Model warmed up!")

    
    def add_documents(self, uploaded_files, use_ocr_tables: bool = False) -> List[Dict]:
        """Add and process uploaded PDFs
        
        Args:
            uploaded_files: List of uploaded PDF files
            use_ocr_tables: If True, enable Table Mode - uses OCR + LLM to extract and structure tables properly.
        """
        results = []

        # Filter out already-uploaded files (by filename) to avoid double-counting and re-processing
        existing = {doc['filename'] for doc in self.documents}
        new_files = [f for f in uploaded_files if getattr(f, 'name', '') not in existing]

        # Report duplicates as skipped
        for dup in uploaded_files:
            if dup not in new_files:
                results.append({
                    'success': False,
                    'filename': getattr(dup, 'name', 'unknown'),
                    'error': 'Already uploaded in this session; skipped'
                })
        uploaded_files = new_files
        
        # Check limits
        if len(self.documents) + len(uploaded_files) > CONFIG['max_pdfs_per_session']:
            results.append({'success': False, 'filename': 'all', 'error': f"Maximum {CONFIG['max_pdfs_per_session']} PDFs per session"})
            return results
        
        # If nothing new to add, return early with duplicate notices
        if not uploaded_files:
            return results

        # Calculate total size
        total_size = sum(doc['size_mb'] for doc in self.documents)
        new_size = sum(file.size for file in uploaded_files) / (1024 * 1024)
        
        if total_size + new_size > CONFIG['max_total_size_mb']:
            return [{'success': False, 'filename': 'all', 'error': f"Total size exceeds {CONFIG['max_total_size_mb']}MB limit"}]
        
        for uploaded_file in uploaded_files:
            try:
                # Save PDF
                pdf_path = self.pdf_dir / uploaded_file.name
                with open(pdf_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Generate comprehensive PDF metadata
                pdf_metadata = generate_pdf_metadata(str(pdf_path))
                
                # Extract text with section tagging
                chunks = PDFProcessor.extract_text(str(pdf_path), use_ocr_tables=use_ocr_tables, session=self)
                
                # Track section distribution
                section_distribution = {}
                for chunk in chunks:
                    section = chunk.get('section_tag', 'GENERAL')
                    section_distribution[section] = section_distribution.get(section, 0) + 1
                
                # Generate embeddings
                texts = [chunk['text'] for chunk in chunks]
                embeddings = self.embedding_model.encode(texts, convert_to_numpy=True).tolist()
                
                # Add to ChromaDB with enhanced metadata
                ids = [f"{uploaded_file.name}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [{
                    'source': chunk['source'], 
                    'page': chunk['page'],
                    'section_tag': chunk.get('section_tag', 'GENERAL')
                } for chunk in chunks]
                
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                
                # Update metadata with comprehensive info
                doc_info = {
                    'filename': uploaded_file.name,
                    'size_mb': uploaded_file.size / (1024 * 1024),
                    'pages': max(chunk['page'] for chunk in chunks),
                    'chunks': len(chunks),
                    'added_at': datetime.now().isoformat(),
                    'pdf_metadata': pdf_metadata,
                    'section_distribution': section_distribution,
                    'keywords': pdf_metadata.get('keywords', []),
                    'has_tables': pdf_metadata.get('has_tables', False)
                }
                self.documents.append(doc_info)
                
                results.append({
                    'success': True,
                    'filename': uploaded_file.name,
                    'chunks': len(chunks)
                })
            
            except Exception as e:
                results.append({
                    'success': False,
                    'filename': uploaded_file.name,
                    'error': str(e)
                })
        
        self._save_metadata()
        return results
    
    def add_document(self, pdf_path, use_ocr_tables: bool = False):
        """Add a single document (wrapper for compatibility).
        
        Args:
            pdf_path: Filesystem path or UploadedFile-like object
            use_ocr_tables: If True, enable Table Mode - uses OCR + LLM to extract and structure tables properly.
        """

        # If already an UploadedFile-like object, pass through
        if hasattr(pdf_path, 'getbuffer') and hasattr(pdf_path, 'name'):
            file_obj = pdf_path
        else:
            # Treat as filesystem path
            class FakeUploadedFile:
                def __init__(self, path):
                    self.name = Path(path).name
                    self.path = path
                    self.size = Path(path).stat().st_size
                
                def getbuffer(self):
                    with open(self.path, 'rb') as f:
                        return f.read()
            file_obj = FakeUploadedFile(pdf_path)
        
        results = self.add_documents([file_obj], use_ocr_tables=use_ocr_tables)
        
        if results and results[0].get('success'):
            return results[0]
        else:
            raise Exception(results[0].get('error', 'Unknown error') if results else 'Failed to add document')
    
    def list_documents(self) -> Dict:
        """List all documents in session"""
        doc_dict = {}
        for doc in self.documents:
            doc_id = doc['filename']
            doc_dict[doc_id] = {
                'filename': doc['filename'],
                'pages': doc.get('pages', 0),
                'chunks': doc.get('chunks', 0),
                'timestamp': doc.get('added_at', ''),
                'size_mb': doc.get('size_mb', 0),
                'keywords': doc.get('keywords', []),
                'has_tables': doc.get('has_tables', False),
                'section_distribution': doc.get('section_distribution', {})
            }
        return doc_dict
    
    def get_document_chunks(self, filename: str) -> List[Dict]:
        """Get all chunks for a specific document with metadata
        
        Args:
            filename: Name of the document
            
        Returns:
            List of chunks with content, page, section_tag, and chunk_id
        """
        try:
            # Query ChromaDB for all chunks from this document
            results = self.collection.get(
                where={'source': filename},
                include=['documents', 'metadatas']
            )
            
            chunks = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    chunk_info = {
                        'chunk_id': i + 1,
                        'content': doc,
                        'page': results['metadatas'][i].get('page', 0),
                        'section_tag': results['metadatas'][i].get('section_tag', 'GENERAL'),
                        'source': results['metadatas'][i].get('source', filename),
                        'preview': doc[:200] + '...' if len(doc) > 200 else doc,
                        'length': len(doc),
                        'word_count': len(doc.split())
                    }
                    chunks.append(chunk_info)
            
            # Sort by page number
            chunks.sort(key=lambda x: (x['page'], x['chunk_id']))
            
            return chunks
        except Exception as e:
            print(f"Error retrieving chunks for {filename}: {e}")
            return []
    
    def delete_document(self, doc_id: str):
        """Delete document by ID (filename)"""
        self.remove_document(doc_id)
    
    def query(self, question: str, compliance_mode: bool = True, selected_document: str = None) -> Tuple[str, Dict]:
        """Query interface (wrapper for compatibility)
        
        Args:
            question: User's question
            compliance_mode: If True, analyze for RA 9184 compliance
            selected_document: Filter chunks to specific document filename (None = all documents)
        """
        return self.generate_response(question, compliance_mode, selected_document)
    
    def load_chat_history(self) -> List[Dict]:
        """Load chat history"""
        return self.chat_history
    
    def save_chat_history(self, history: List[Dict]):
        """Save chat history"""
        self.chat_history = history
        self._save_metadata()
    
    def remove_document(self, filename: str):
        """Remove document from session"""
        # Remove from ChromaDB
        chunk_ids = self.collection.get(where={'source': filename})['ids']
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)
        
        # Remove PDF file
        pdf_path = self.pdf_dir / filename
        if pdf_path.exists():
            pdf_path.unlink()
        
        # Update metadata
        self.documents = [doc for doc in self.documents if doc['filename'] != filename]
        self._save_metadata()
    
    def retrieve_context(self, query: str, top_k: int = None, selected_document: str = None) -> List[Dict]:
        """Retrieve relevant chunks using hybrid scoring (semantic + lightweight enhancements)
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            selected_document: Filter mode - 'General' (knowledge base only), None (all documents), or filename (specific file)
        """
        if top_k is None:
            top_k = CONFIG['top_k_chunks']
        
        all_chunks = []
        
        # Check if collection exists
        if self.collection is None:
            print(f"⚠️ No documents uploaded")
            return all_chunks
        
        # GENERAL MODE: Only retrieve from knowledge base (permanent_knowledge type)
        if selected_document == "General":
            print(f"🔍 General mode: Retrieving only from knowledge base (no uploaded documents)")
            kb_collection = self._get_knowledge_base()
            
            if kb_collection is None:
                print("⚠️ Knowledge base not available. Run: python knowledge_base_setup.py")
                return []  # No knowledge base available
            
            # For knowledge base, get more chunks for better coverage
            kb_top_k = min(top_k, 5)  # Get up to 5 knowledge base chunks
            print(f"📚 Retrieving up to {kb_top_k} knowledge base chunks")
            
            # Query knowledge base
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
            
            try:
                kb_results = kb_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=kb_top_k
                )
                
                # Format knowledge base results
                kb_chunks = []
                if kb_results['documents'] and kb_results['documents'][0]:
                    for i, doc in enumerate(kb_results['documents'][0]):
                        distance = kb_results['distances'][0][i]
                        cosine_score = 1 - distance
                        
                        kb_chunks.append({
                            'content': doc,
                            'score': cosine_score,  # Use cosine as final score for KB
                            'cosine_score': cosine_score,
                            'source': kb_results['metadatas'][0][i].get('source', 'Knowledge Base'),
                            'page': 'N/A',
                            'section_tag': kb_results['metadatas'][0][i].get('topic', 'GENERAL'),
                            'type': 'permanent_knowledge',
                            'category': kb_results['metadatas'][0][i].get('category', 'General'),
                            'topic': kb_results['metadatas'][0][i].get('topic', 'General')
                        })
                
                if kb_chunks:
                    print(f"📚 Retrieved {len(kb_chunks)} chunks from knowledge base")
                else:
                    print("⚠️ No matching chunks found in knowledge base")
                return kb_chunks
                
            except Exception as e:
                print(f"Knowledge base query error: {e}")
                return []
        
        # Get more candidates for reranking to improve recall
        candidate_k = max(top_k * 3, top_k + 5)
        
        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True).tolist()
        
        # Build where filter for document selection with SESSION ISOLATION
        where_filter = None
        
        if selected_document and selected_document not in ["All Documents", "General"]:
            # Specific file selected - STRICT filtering
            where_filter = {"source": selected_document}
            print(f"🔍 STRICT FILTER: Only retrieving from document '{selected_document}'")
            print(f"🔍 Available documents: {[doc['filename'] for doc in self.documents]}")
        elif selected_document is None or selected_document == "All Documents":
            # All Documents mode - ONLY from current session's files
            session_filenames = [doc['filename'] for doc in self.documents]
            if session_filenames:
                # Use $in operator to filter to only session files
                where_filter = {"source": {"$in": session_filenames}}
                print(f"🔍 All Documents mode: Retrieving from {len(session_filenames)} files in current session")
            else:
                print("⚠️ No documents in current session")
                return []
        
        # Search ChromaDB (primary semantic search)
        try:
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": candidate_k
            }
            
            # Add where filter if specific document selected
            if where_filter:
                query_params["where"] = where_filter
            
            results = self.collection.query(**query_params)
            
            # VERIFY: Post-retrieval check that all chunks match selected document
            if selected_document and selected_document not in ["All Documents", "General"]:
                if results['metadatas'] and results['metadatas'][0]:
                    chunk_sources = [meta.get('source', 'unknown') for meta in results['metadatas'][0]]
                    mismatched = [src for src in chunk_sources if src != selected_document]
                    if mismatched:
                        print(f"⚠️ WARNING: Found {len(mismatched)} chunks from wrong documents: {set(mismatched)}")
                        print(f"⚠️ Expected only: '{selected_document}'")
                    else:
                        print(f"✅ VERIFIED: All {len(chunk_sources)} chunks are from '{selected_document}'")
        except Exception as e:
            print(f"ChromaDB query error: {e}")
            return all_chunks  # Return only permanent knowledge if document search fails
        
        chunks = []
        if results['documents']:
            selected_indices = []
            
            for i, doc in enumerate(results['documents'][0]):
                # 1. Cosine similarity (primary score)
                distance = results['distances'][0][i]
                cosine_score = 1 - distance
                
                # Debug: Print distance and cosine score for first few chunks
                if i < 3:
                    print(f"🔍 DEBUG Chunk {i}: distance={distance:.4f}, cosine={cosine_score:.4f}")
                
                # 2. LLM Judge score (lightweight: based on query-chunk overlap)
                query_words = set(query.lower().split())
                chunk_words = set(doc.lower().split())
                word_overlap = len(query_words & chunk_words) / max(len(query_words), 1)
                llm_judge_score = min(word_overlap * 2, 1.0)  # Normalize to [0,1]
                
                # 3. Structural weight (section tag relevance)
                section_tag = results['metadatas'][0][i].get('section_tag', 'GENERAL')
                structural_score = 1.0  # Neutral (no section bias)
                
                # 4. Metadata score (simple document recency/page position)
                metadata_score = 1.0  # Neutral (no page bias)
                
                # 5. MMR score (diversity)
                mmr_score = calculate_mmr_score(query_embedding, None, selected_indices)
                
                # Weighted hybrid score (subtle enhancement, cosine dominant)
                weights = CONFIG['retrieval_weights']
                final_score = (
                    weights['cosine'] * cosine_score +
                    weights['llm_judge'] * llm_judge_score +
                    weights['structural'] * structural_score +
                    weights['metadata'] * metadata_score +
                    weights['mmr'] * mmr_score
                )
                
                # Get source filename for logging
                source_filename = results['metadatas'][0][i]['source']
                
                # Document boost
                document_boost = 1.0
                
                chunks.append({
                    'content': doc,
                    'score': final_score * document_boost,
                    'cosine_score': cosine_score,
                    'source': source_filename,
                    'page': results['metadatas'][0][i]['page'],
                    'section_tag': section_tag,
                    'type': 'document'
                })
                
                selected_indices.append(i)
        
        # Sort document chunks by score
        all_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)[:top_k]
        
        return all_chunks
    
    def generate_response(self, question: str, compliance_mode: bool = True, selected_document: str = None) -> Tuple[str, Dict]:
        """Generate response using RAG
        
        Args:
            question: User's question
            compliance_mode: If True, analyze for RA 9184 compliance; if False, just answer questions
            selected_document: Filter chunks to specific document filename (None = all documents)
        """
        import re  # Import here to avoid module-level issues
        start_time = time.time()
        
        # Check if collection is initialized (only needed for document queries)
        # Allow queries even without documents if using knowledge base
        has_documents = len(self.documents) > 0
        if not has_documents and self.collection is None:
            # No documents and no KB - can't answer
            pass  # Will be handled by context retrieval
        
        # DETECT KNOWLEDGE BASE / GENERAL INFORMATION QUERIES
        # For "what is", "explain", "tell me about" questions, force non-compliance mode for detailed answers
        q_lower = question.lower().strip()
        is_general_info_query = any(kw in q_lower for kw in [
            'what is', 'what are', 'explain', 'tell me about', 'describe', 'definition of', 'meaning of'
        ])
        
        # Override compliance mode for general information queries
        if is_general_info_query and selected_document == "General":
            compliance_mode = False  # Force detailed response
            print("🔍 General information query detected - using detailed response mode")
        
        # COMPLIANCE MODE: Use RAG retrieval + special compliance prompt
        compliance_issues = []
        # Only trigger compliance workflow on explicit compliance checks (not general "what is" questions)
        is_compliance_query = any(kw in q_lower for kw in [
            'check compliance', 'verify compliance', 'compliance check', 'compliance checking', 'compliance report', 'missing fields', 'non-compliant', 'compliance issues'
        ])
        # If user is asking what RA 9184 is (or any "what is" question), skip compliance path
        if is_general_info_query:
            is_compliance_query = False
        
        # For compliance checking, require a selected document
        if is_compliance_query and (not selected_document or selected_document in ["All Documents", "General"]):
            return "⚠️ Please select a specific document to check compliance. You cannot check compliance for 'All Documents'.", {}
        
        if compliance_mode and is_compliance_query:
            # Use RAG retrieval to get relevant chunks
            retrieval_start = time.time()
            # Retrieve chunks about compliance fields
            compliance_query = "ABC Approved Budget Cost PR Number Purchase Request Delivery Period Pre-Bid Conference Bid Opening Date Closing Date"
            print(f"📄 COMPLIANCE CHECK: Document filter = '{selected_document}'")
            print(f"🔍 COMPLIANCE CHECK: Available documents = {[doc['filename'] for doc in self.documents]}")
            chunks = self.retrieve_context(compliance_query, top_k=10, selected_document=selected_document)
            print(f"✅ COMPLIANCE CHECK: Retrieved {len(chunks)} chunks")
            retrieval_time = time.time() - retrieval_start
            
            if not chunks:
                return "⚠️ Could not retrieve document content for compliance checking.", {}
            
            # Build context from chunks
            context = "\n\n".join([chunk['content'] for chunk in chunks])
            
            # Build compliance analysis prompt
            compliance_prompt = f"""Analyze this procurement document for RA 9184 compliance. Extract these 6 required fields:

DOCUMENT CONTENT:
{context}

Extract and list each field (write EXACT value from document or "MISSING" if not found):

1. ABC (Approved Budget Cost/Contract): 
2. PR Number (Purchase Request Number): 
3. Delivery Period: 
4. Pre-Bid Conference (date, time, location): 
5. Bid Opening Date (date, time): 
6. Closing Date: 

Then provide a brief compliance assessment explaining whether the document follows RA 9184 requirements."""
            
            # Generate response
            generation_start = time.time()
            try:
                inputs = self._llm_tokenizer(compliance_prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self._llm_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=self._llm_tokenizer.eos_token_id
                    )
                
                response = self._llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove prompt echo
                if compliance_prompt in response:
                    response = response.replace(compliance_prompt, "").strip()
                
                generation_time = time.time() - generation_start
                
                # Build debug info
                debug_info = {
                    'retrieval_time': retrieval_time,
                    'generation_time': generation_time,
                    'total_time': retrieval_time + generation_time,
                    'chunks': [{
                        'content': chunk['content'],  # Full content for preview
                        'source': chunk['source'],
                        'page': chunk['page'],
                        'score': chunk['score'],
                        'type': chunk.get('type', 'document')
                    } for chunk in chunks],
                    'session_id': self.session_id,
                    'session_name': self.session_name,
                    'collection_name': f"session_{self.session_id}",
                    'input_tokens': len(self._llm_tokenizer.encode(compliance_prompt)),
                    'output_tokens': len(self._llm_tokenizer.encode(response)),
                    'total_tokens': len(self._llm_tokenizer.encode(compliance_prompt)) + len(self._llm_tokenizer.encode(response))
                }
                
                return response, debug_info
                
            except Exception as e:
                print(f"Error during compliance check: {e}")
                import traceback
                traceback.print_exc()
                return f"⚠️ Compliance check failed: {str(e)}", {}
        
        # Retrieve context
        retrieval_start = time.time()
        chunks = self.retrieve_context(question, selected_document=selected_document)
        retrieval_time = time.time() - retrieval_start
        
        # DETECT VAGUE QUERIES when viewing All Documents
        # Check if query is too generic and multiple documents are loaded
        has_multiple_docs = len(self.documents) > 1
        viewing_all_docs = selected_document is None or selected_document == "All Documents"
        
        # Vague query patterns (asking about "the document" when multiple exist)
        vague_patterns = [
            r'\b(?:what|what\'s|whats)\s+(?:is\s+)?(?:this|the)\s+document\s+(?:about|for)',
            r'\b(?:describe|explain)\s+(?:this|the)\s+document',
            r'\btell\s+me\s+about\s+(?:this|the)\s+document',
            r'\b(?:what|what\'s|whats)\s+(?:in\s+)?(?:this|the)\s+document',
            r'\bdocument\s+(?:title|name|subject)',
        ]
        
        is_vague_query = any(re.search(pattern, question.lower()) for pattern in vague_patterns)
        
        if has_multiple_docs and viewing_all_docs and is_vague_query:
            # List available documents
            doc_list = "\n".join([f"• {doc['filename']}" for doc in self.documents])
            
            vague_response = f"""⚠️ **Your question is too vague** - you're viewing **All Documents** ({len(self.documents)} documents loaded), but asking about "the document" without specifying which one.

**Available documents:**
{doc_list}

**Please either:**
1. Select a specific document from the dropdown above, OR
2. Ask a more specific question (e.g., "What is the ABC for the WMSU project?")

This helps me provide accurate answers instead of mixing information from different documents."""
            
            debug_info = {
                'session_id': self.session_id,
                'session_name': self.session_name,
                'collection_name': f"session_{self.session_id}",
                'chunks': [],
                'scores': [],
                'retrieval_time': retrieval_time,
                'generation_time': 0,
                'response_time': time.time() - start_time,
                'document_chunks_retrieved': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'from_cache': False,
                'judge_confidence': 40,  # Low confidence for vague queries
                'vague_query_detected': True
            }
            return vague_response, debug_info
        
        if not chunks:
            # If no chunks found, provide debug info explaining why
            has_docs = len(self.documents) > 0
            is_asking_doc = any(kw in question.lower() for kw in ['this document', 'the document', 'uploaded'])
            
            if is_asking_doc and not has_docs:
                error_msg = "📄 You're asking about a document, but no documents have been uploaded yet. Please upload documents in the Documents tab first."
            elif not has_docs:
                error_msg = "📁 No documents uploaded. Please upload documents in the Documents tab to get started."
            else:
                error_msg = f"🔍 No relevant chunks found in {len(self.documents)} uploaded document(s). Try rephrasing your question or check if the information exists in the documents."
            
            debug_info = {
                'session_id': self.session_id,
                'session_name': self.session_name,
                'collection_name': f"session_{self.session_id}",
                'chunks': [],
                'scores': [],
                'retrieval_time': retrieval_time,
                'generation_time': 0,
                'response_time': time.time() - start_time,
                'document_chunks_retrieved': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'from_cache': False,
                'error': error_msg
            }
            return error_msg, debug_info
        
        # Build context with source indicators
        context_parts = []
        for chunk in chunks:
            if chunk.get('type') == 'permanent_knowledge':
                # For KB, use simpler format - just the content
                context_parts.append(chunk['content'])
            else:
                # Regular document chunks
                context_parts.append(f"[{chunk['source']}, p.{chunk['page']}] {chunk['content']}")
        
        context = "\n\n".join(context_parts)

        # Create debug_info early (before any early returns)
        debug_info = {
            'session_id': self.session_id,
            'session_name': self.session_name,
            'collection_name': f"session_{self.session_id}",
            'chunks': [
                {
                    'content': c.get('content', ''),
                    'type': c.get('type', 'document'),
                    'score': c.get('score', 0),
                    'cosine_score': c.get('cosine_score', 0),
                    'source': c.get('source', 'Unknown'),
                    'page': c.get('page', 'N/A'),
                    'section_tag': c.get('section_tag', '')
                }
                for c in chunks
            ],
            'scores': [c['score'] for c in chunks],
            'retrieval_time': retrieval_time,
            'document_chunks_retrieved': len(chunks)
        }

        def finalize_early(response_text: str):
            generation_time = 0.0
            debug_info['response_time'] = max(time.time() - start_time, 0.001)
            debug_info['generation_time'] = generation_time
            if hasattr(self, '_llm_tokenizer') and self._llm_tokenizer:
                debug_info['input_tokens'] = len(self._llm_tokenizer.encode(context + question))
                debug_info['output_tokens'] = len(self._llm_tokenizer.encode(response_text))
            else:
                debug_info['input_tokens'] = None
                debug_info['output_tokens'] = None
            return response_text, debug_info

        
        # Load model from global cache if not already loaded
        if self._llm_model is None:
            self.load_llm()
        
        generation_start = time.time()
        
        # Use chat template format if available (better for instruction following)
        if hasattr(self._llm_tokenizer, 'chat_template') and self._llm_tokenizer.chat_template:
            # Format as chat messages
            if compliance_mode and is_compliance_query:
                # Add RA 9184 compliance instructions and rule-based findings
                compliance_findings = "\n".join(compliance_issues) if compliance_issues else "No automated checks completed yet."
                
                ra9184_instructions = """RA 9184 COMPLIANCE REQUIREMENTS:

Republic Act 9184 (Government Procurement Reform Act) requires procurement documents to include:

1. **ABC (Approved Budget for the Contract)** - The total budget allocated for the procurement
2. **Purchase Request (PR) Number** - Reference number for the procurement request
3. **Delivery Period** - Timeframe for delivery of goods/services
4. **Pre-Bid Conference Date** - When bidders can ask questions
5. **Bid Opening Date** - Deadline for bid submission and opening
6. **Closing Date** - Final deadline for the procurement process
7. **Complete Bidding Schedule** - All dates and milestones
8. **Scope of Work** - Clear description of requirements
9. **Eligibility Requirements** - Legal and technical requirements for bidders
10. **Evaluation Criteria** - How bids will be evaluated

Your task: Review the document excerpts below and identify which required fields are present and which are missing."""
                
                messages = [
                    {"role": "user", "content": f"""{ra9184_instructions}

AUTOMATED COMPLIANCE FINDINGS (rule-based analysis):
{compliance_findings}

DOCUMENT CONTENT:
{context}

Question: {question}

Provide a clear compliance report listing what was found and what is missing."""}
                ]
            elif compliance_mode:
                messages = [
                    {"role": "user", "content": f"""Answer the question based on the document content below.

DOCUMENT CONTENT:
{context}

Question: {question}

Answer:"""}
                ]
            else:
                # For general questions, allow detailed responses
                # Check if user wants detailed explanation
                wants_detail = any(kw in question.lower() for kw in ['explain', 'detail', 'comprehensive', 'describe', 'tell me more', 'in depth'])
                
                # SIMPLIFIED PROMPT FOR SMALL MODEL - especially for knowledge base
                is_kb_query = chunks and chunks[0].get('type') == 'permanent_knowledge'
                
                if is_kb_query or wants_detail:
                    # For knowledge base OR detailed questions: Use comprehensive prompt
                    messages = [
                        {"role": "user", "content": f"""You are a helpful assistant providing detailed, comprehensive answers about Philippine government procurement regulations.

REFERENCE INFORMATION:
{context}

USER QUESTION: {question}

INSTRUCTIONS: Provide a complete, detailed answer using ALL relevant information from the reference material above. Include specific details, dates, requirements, procedures, and examples. Your answer should be thorough and comprehensive."""}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": f"""Information:
{context}

Question: {question}

Answer:"""}
                    ]
            
            # Apply chat template
            formatted_prompt = self._llm_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = self._llm_tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
        else:
            # Fallback to simple format without prompt structure
            if compliance_mode and is_compliance_query:
                ra9184_instructions = """RA 9184 COMPLIANCE REQUIREMENTS:
Republic Act 9184 requires: ABC (budget), PR Number, Delivery Period, Pre-Bid Conference Date, Bid Opening Date, Closing Date, Complete Schedule, Scope of Work, Eligibility Requirements, and Evaluation Criteria."""
                
                simple_prompt = f"""{ra9184_instructions}

Document Content:
{context}

TASK: Check if these required fields exist in the document:
1. ABC (Approved Budget for Contract)
2. PR Number (Purchase Request)
3. Delivery Period
4. Pre-Bid Conference Date
5. Bid Opening Date
6. Closing Date
7. Complete Bidding Schedule

Question: {question}

List FOUND items with ✓ and MISSING items with ❌:"""
            elif compliance_mode:
                simple_prompt = f"""Document Content:
{context}

Question: {question}

Answer based on the document:"""
            else:
                simple_prompt = f"""Based on the following document, provide a detailed answer to the question.

DOCUMENT:
{context}

QUESTION: {question}

ANSWER (provide comprehensive details):"""
            
            inputs = self._llm_tokenizer(simple_prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move inputs to same device as model
        device = next(self._llm_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Prepare generation parameters - optimized for deterministic output
        # Use higher token limit for General/knowledge base queries (detailed answers)
        is_kb_or_general = selected_document == "General" or (chunks and chunks[0].get('type') == 'permanent_knowledge')
        max_tokens = 1024 if is_kb_or_general else CONFIG['max_new_tokens']  # 1024 for General, 512 for documents
        
        gen_kwargs = {
            'max_new_tokens': max_tokens,
            'pad_token_id': self._llm_tokenizer.eos_token_id,
        }
        
        # Add sampling parameters only when enabled
        if CONFIG['do_sample']:
            gen_kwargs['do_sample'] = True
            gen_kwargs['temperature'] = CONFIG['temperature']
            gen_kwargs['top_p'] = 0.9
            gen_kwargs['top_k'] = 50
        else:
            # For greedy decoding, explicitly set do_sample=False
            gen_kwargs['do_sample'] = False
        
        with torch.no_grad():
            outputs = self._llm_model.generate(**inputs, **gen_kwargs)
        
        # Decode response
        full_response = self._llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated answer (remove input prompt)
        input_text = self._llm_tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
        
        # Debug: Print what we're getting
        print(f"\n🔍 DEBUG Generation:")
        print(f"  Full response length: {len(full_response)}")
        print(f"  Input text length: {len(input_text)}")
        print(f"  Full response preview: {full_response[:200]}...")
        
        if full_response.startswith(input_text):
            response = full_response[len(input_text):].strip()
        else:
            response = full_response
        
        print(f"  Extracted response length: {len(response)}")
        print(f"  Extracted response: {response[:500]}")
        
        # If response is empty or too short, there might be a template issue
        if not response or len(response) < 10:
            # Try splitting on common delimiters
            if "assistant" in full_response.lower():
                parts = full_response.split("assistant")
                if len(parts) > 1:
                    response = parts[-1].strip()
            elif "Answer:" in full_response:
                response = full_response.split("Answer:")[-1].strip()
        
        # Remove any remaining prompt artifacts
        response = response.strip()
        
        # Clean up common artifacts but be LESS aggressive
        for artifact in ["CONTEXT:", "DOCUMENT CONTENT:", "Question:", "Answer:"]:
            if response.startswith(artifact):
                response = response[len(artifact):].strip()
        
        # Final cleanup - remove leading/trailing markers
        response = response.strip('\n "*')
        
        print(f"  Final cleaned response length: {len(response)}")
        print(f"  Final response preview: {response[:200]}")
        
        # FALLBACK: Only if response is completely empty (LLM failed to generate)
        if not response or len(response) < 5:
            # Prefer document chunks over knowledge base for factual answers
            doc_chunks = [c for c in chunks if c.get('type') == 'document']
            kb_chunks = [c for c in chunks if c.get('type') == 'permanent_knowledge']
            
            if doc_chunks and len(doc_chunks) > 0:
                # Return the most relevant document chunk directly
                response = f"Based on the document:\n\n{doc_chunks[0]['content'][:500]}"
            elif kb_chunks and len(kb_chunks) > 0:
                # Fallback to knowledge base only if no document chunks
                response = f"**{kb_chunks[0].get('topic', 'Information').replace('_', ' ').title()}**\n\n"
                response += kb_chunks[0]['content']
                if len(kb_chunks) > 1:
                    response += f"\n\n**Related Information:**\n{kb_chunks[1]['content'][:300]}..."
        
        generation_time = time.time() - generation_start
        
        # LLM-as-a-Judge validation - ONLY for "All Documents" mode
        judge_result = None
        # Only run judge when user is viewing All Documents (to catch mis-retrieval)
        # Skip judge for single document or General (knowledge base) selections
        should_use_judge = (
            CONFIG['judge_enabled'] and 
            response and 
            len(response) >= 5 and
            (selected_document is None or selected_document == "All Documents")
        )
        
        if should_use_judge:
            print(f"\n⚖️ Running LLM Judge validation (All Documents mode)...")
            judge_start = time.time()
            judge_result = self._judge_response(question, response, chunks)
            judge_time = time.time() - judge_start
            
            print(f"  Judge confidence: {judge_result['confidence']}%")
            print(f"  Judge reasoning: {judge_result['reasoning'][:100]}...")
            print(f"  Judge time: {judge_time:.2f}s")
            
            debug_info['judge_confidence'] = judge_result['confidence']
            debug_info['judge_reasoning'] = judge_result['reasoning']
            debug_info['judge_time'] = judge_time
            debug_info['judge_issues'] = judge_result.get('issues', [])
        
        # Update debug info with generation metrics
        debug_info['response_time'] = max(time.time() - start_time, 0.001)
        debug_info['retrieval_time'] = retrieval_time
        debug_info['generation_time'] = generation_time
        debug_info['input_tokens'] = len(self._llm_tokenizer.encode(context + question)) if hasattr(self, '_llm_tokenizer') and self._llm_tokenizer else None
        debug_info['output_tokens'] = len(self._llm_tokenizer.encode(response)) if hasattr(self, '_llm_tokenizer') and self._llm_tokenizer else None
        debug_info['from_cache'] = False
        
        return response, debug_info
    
    def _judge_response(self, question: str, response: str, chunks: List[Dict]) -> Dict:
        """
        Use base LLM (without adapter) to evaluate response quality
        
        Returns:
            Dict with confidence score (0-100), reasoning, and issues
        """
        try:
            # Ensure base model is loaded
            if self._base_model is None:
                # Try to get from cache
                if HAS_STREAMLIT and st is not None:
                    try:
                        cached_data = get_cached_llm_model()
                        self._base_model = cached_data.get('base_model')
                    except:
                        pass
                
                if self._base_model is None:
                    # Judge not available
                    return {
                        'confidence': 75,  # Default confidence
                        'reasoning': 'Judge model not available',
                        'issues': []
                    }
            
            # Prepare context from chunks WITH document metadata
            context_parts = []
            doc_names = []
            for i, chunk in enumerate(chunks[:3]):
                doc_name = chunk.get('source', 'Unknown')
                doc_names.append(doc_name)
                page_num = chunk.get('page', '?')
                content = chunk['content'][:250]  # Slightly shorter for better focus
                context_parts.append(f"[Document #{i+1}: {doc_name}, Page {page_num}]\n{content}...")
            
            context_text = "\n\n".join(context_parts)
            unique_docs = list(set(doc_names))
            
            # Simplified, more direct judge prompt
            judge_prompt = f"""Check if the answer is correct.

QUESTION: {question}

DOCUMENTS RETRIEVED:
{', '.join(unique_docs)}

CONTENT FROM DOCUMENTS:
{context_text}

ANSWER: {response}

CRITICAL CHECK - Does the question ask about a specific document/project, and if so, do the retrieved documents match?
- Question asks about: Look for specific project/document names in the question
- Retrieved documents are: {', '.join(unique_docs)}
- Match? If question mentions a specific document but retrieved docs are different topics, score 0-35

EVALUATION:
Score 0-100 where:
- 0-35: WRONG DOCUMENTS (question asks about X but got documents about Y) OR completely wrong answer
- 40-55: Vague question OR documents partially relevant but answer has issues  
- 60-80: Documents match but answer incomplete or could be better
- 85-100: Documents match what question asks about AND answer is correct

Output format:
CONFIDENCE: [number]
REASONING: [Is this the right document? Is answer correct?]"""
            
            # Tokenize
            inputs = self._llm_tokenizer(
                judge_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1500
            ).to(self._device)
            
            # Generate with fine-tuned model (better quality than base)
            with torch.no_grad():
                outputs = self._llm_model.generate(
                    **inputs,
                    max_new_tokens=CONFIG['judge_max_tokens'],
                    temperature=CONFIG['judge_temperature'],
                    do_sample=False,  # Greedy for consistency
                    pad_token_id=self._llm_tokenizer.pad_token_id
                )
            
            # Decode
            full_judge_response = self._llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract judge output (remove prompt)
            if judge_prompt in full_judge_response:
                judge_output = full_judge_response.split(judge_prompt)[-1].strip()
            else:
                judge_output = full_judge_response
            
            print(f"\n  🔍 Judge raw output:\n{judge_output[:300]}...")
            
            # Improved parsing with regex for robustness
            confidence = 50  # Default
            reasoning = ""
            issues = []
            
            # Extract CONFIDENCE with flexible matching
            conf_match = re.search(r'CONFIDENCE[:\s]+(\d+)', judge_output, re.IGNORECASE)
            if conf_match:
                confidence = min(100, max(0, int(conf_match.group(1))))
            else:
                # Fallback: look for standalone number 0-100
                num_match = re.search(r'\b([0-9]{1,3})\b', judge_output)
                if num_match:
                    num = int(num_match.group(1))
                    if 0 <= num <= 100:
                        confidence = num
                else:
                    # LLM judge failed to parse - apply safety checks
                    # CRITICAL: Check if question asks for specific document but got different docs
                    
                    # Extract document names from chunks
                    doc_names = [chunk.get('source', '') for chunk in chunks[:3]]
                    unique_docs = list(set(doc_names))
                    
                    # Get the TOP-RANKED document (first chunk is highest score)
                    top_doc = chunks[0].get('source', '') if chunks else ''
                    
                    # Check for vague questions first
                    vague_patterns = [
                        r'^\s*what\s*\??\s*$',  # Just "what?" or "what"
                        r'^\s*what\s+is\s+this\s*\??\s*$',  # "what is this?"
                        r'^\s*what\s+about\s*\??\s*$',  # "what about?"
                        r'^\s*tell\s+me\s*\??\s*$',  # "tell me?"
                        r'^\s*huh\s*\??\s*$',  # "huh?"
                        r'^\s*explain\s*\??\s*$',  # Just "explain?"
                    ]
                    
                    is_vague = any(re.match(pattern, question.lower().strip()) for pattern in vague_patterns)
                    
                    if is_vague:
                        confidence = 45  # Low confidence for vague questions
                        reasoning = "Question is too vague or unclear - needs more specificity"
                    else:
                        # SAFETY CHECK: Look for document/project keywords in question
                        # Extract potential document keywords from question (capitalized words, multiple words)
                        all_keywords = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', question)
                        
                        # Filter out common stopwords that appear in questions (not document-specific)
                        stopwords = {
                            'What', 'When', 'Where', 'Who', 'Which', 'How', 'Why',
                            'The', 'For', 'And', 'But', 'Or', 'Of', 'To', 'In', 'On', 'At',
                            'Approved', 'Budget', 'Contract', 'ABC',  # Common in ALL procurement docs
                            'Date', 'Time', 'Schedule', 'Information', 'Details',
                            'Please', 'Tell', 'Me', 'About', 'Provide', 'Give'
                        }
                        
                        # Keep only document-specific keywords (filter out stopwords)
                        question_keywords = [kw for kw in all_keywords if kw not in stopwords]
                        
                        # If no specific keywords after filtering, use original list but mark as uncertain
                        if not question_keywords and all_keywords:
                            question_keywords = all_keywords
                            is_generic_question = True
                        else:
                            is_generic_question = False
                        
                        # Check if ANY of the retrieved documents match question keywords
                        # (LLM can use any of the top-k chunks to answer)
                        best_match_ratio = 0
                        best_matched_keywords = []
                        best_mismatched_keywords = []
                        best_matching_doc = ""
                        
                        if question_keywords:
                            for doc in unique_docs:
                                matched_keywords = []
                                mismatched_keywords = []
                                
                                for keyword in question_keywords:
                                    if keyword.lower() in doc.lower():
                                        matched_keywords.append(keyword)
                                    else:
                                        mismatched_keywords.append(keyword)
                                
                                # Calculate match ratio for this document
                                match_ratio = len(matched_keywords) / len(question_keywords) if question_keywords else 0
                                
                                # Keep track of best matching document
                                if match_ratio > best_match_ratio:
                                    best_match_ratio = match_ratio
                                    best_matched_keywords = matched_keywords
                                    best_mismatched_keywords = mismatched_keywords
                                    best_matching_doc = doc
                        
                        # STRICTER CHECK: If question has specific terms, verify at least ONE document matches
                        if question_keywords and len(question_keywords) >= 2:
                            if best_match_ratio < 0.3:  # Less than 30% of keywords match ANY document
                                confidence = 40  # Low confidence - likely wrong document retrieval
                                reasoning = f"⚠️ SUSPICIOUS: Question asks about '{', '.join(question_keywords[:3])}' but retrieved documents don't match well (best: '{best_matching_doc[:50]}...'). Verify answer carefully!"
                            elif best_match_ratio < 0.5:  # 30-50% match - partial match
                                confidence = 70  # Medium-good confidence (was 60, increased tolerance)
                                reasoning = f"Judge parsing failed. Found document matching some keywords ('{', '.join(best_matched_keywords[:2])}'). Verify answer against source."
                            else:  # 50%+ match - good match found
                                confidence = 75  # Good confidence
                                reasoning = f"Judge parsing failed, but found relevant document matching '{', '.join(best_matched_keywords[:3])}'. Answer likely valid but verify accuracy."
                        else:
                            confidence = 70  # Neutral confidence when judge parsing fails
                            reasoning = "Unable to parse judge response - answer may be valid but needs verification"
            
            # Extract ISSUES
            issues_match = re.search(r'ISSUES[:\s]+(.+?)(?:REASONING|$)', judge_output, re.IGNORECASE | re.DOTALL)
            if issues_match:
                issues_text = issues_match.group(1).strip()
                if issues_text.lower() not in ['none', 'n/a', 'no issues', '']:
                    issues = [issues_text]
            
            # Extract REASONING
            reasoning_match = re.search(r'REASONING[:\s]+(.+)', judge_output, re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()[:200]  # Limit length
            
            # Fallback if no structured output found
            if not reasoning:
                sentences = [s.strip() for s in judge_output.split('.') if len(s.strip()) > 15]
                reasoning = sentences[0] if sentences else "Answer appears valid based on context"
            
            return {
                'confidence': confidence,
                'reasoning': reasoning,
                'issues': issues
            }
        
        except Exception as e:
            print(f"  ⚠️ Judge evaluation failed: {str(e)}")
            return {
                'confidence': 50,
                'reasoning': f'Judge error: {str(e)}',
                'issues': ['Judge system error']
            }
    
    def add_message(self, role: str, content: str, debug_info: Dict = None):
        """Add message to chat history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        if debug_info:
            message['debug_info'] = debug_info
        
        self.chat_history.append(message)
        self._save_metadata()
    
    def clear_chat_history(self):
        """Clear chat history"""
        self.chat_history = []
        self._save_metadata()
    
    def get_total_chunks(self) -> int:
        """Get total number of chunks in collection"""
        return self.collection.count()
    
    def reload_models(self):
        """Reload models (clear cache)"""
        self._embedding_model = None
        self._llm_model = None
        self._llm_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SessionManager:
    """Manage multiple RAG sessions"""
    
    def __init__(self, sessions_dir: str = "./sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.sessions: Dict[str, RAGSession] = {}
        
        # Load existing sessions
        self._load_sessions()
    
    def _load_sessions(self):
        """Load existing sessions from disk"""
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
                # Check if marked for deletion
                delete_marker = session_dir / '.delete_on_restart'
                if delete_marker.exists():
                    # Try to delete this session
                    try:
                        import subprocess
                        subprocess.run(
                            ['cmd', '/c', 'rmdir', '/s', '/q', str(session_dir)],
                            capture_output=True,
                            timeout=5,
                            check=False
                        )
                        if not session_dir.exists():
                            continue
                    except:
                        pass
                
                metadata_file = session_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        session = RAGSession(
                            session_id=metadata['session_id'],
                            session_name=metadata['session_name'],
                            session_dir=session_dir
                        )
                        self.sessions[metadata['session_id']] = session
                    except Exception as e:
                        print(f"Warning: Could not load session from {session_dir}: {e}")
    
    def create_session(self, session_name: str) -> str:
        """Create new session"""
        # Check limit
        if len(self.sessions) >= CONFIG['max_sessions']:
            # Delete oldest session
            oldest_session = min(
                self.sessions.values(),
                key=lambda s: s.metadata.get('updated_at', s.metadata.get('created_at'))
            )
            self.delete_session(oldest_session.session_id)
        
        # Generate session ID
        session_id = hashlib.md5(
            f"{session_name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Create session
        session_dir = self.sessions_dir / session_id
        session = RAGSession(session_id, session_name, session_dir)
        self.sessions[session_id] = session
        session._save_metadata()
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[RAGSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str):
        """Delete session with proper cleanup and error handling"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            try:
                # Explicitly close the session first to release locks
                session.close()
                
                # Wait a moment for file handles to be released
                import time
                time.sleep(0.5)
                
                # Close ChromaDB client to release file locks
                if hasattr(session, '_chroma_client') and session._chroma_client is not None:
                    try:
                        # Delete collection first
                        if hasattr(session, '_collection') and session._collection is not None:
                            try:
                                session._chroma_client.delete_collection(session._collection.name)
                            except:
                                pass
                        # Clear references
                        session._collection = None
                        session._chroma_client = None
                    except Exception as e:
                        print(f"Warning: Error closing ChromaDB: {e}")
                
                # Clear model references to free memory
                if hasattr(session, '_llm_model'):
                    session._llm_model = None
                if hasattr(session, '_llm_tokenizer'):
                    session._llm_tokenizer = None
                if hasattr(session, '_embedding_model'):
                    session._embedding_model = None
                
                # Force garbage collection
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Wait for file handles to release
                import time
                time.sleep(0.5)
                
                # Remove from memory first
                session_dir = session.session_dir
                del self.sessions[session_id]
                
                # Force another GC after removing from dict
                gc.collect()
                time.sleep(0.3)
                
                # Try to delete directory with retry and better error handling
                if session_dir.exists():
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            # On Windows, use more aggressive deletion
                            import subprocess
                            try:
                                # Try rmdir /s /q for Windows
                                subprocess.run(
                                    ['cmd', '/c', 'rmdir', '/s', '/q', str(session_dir)],
                                    capture_output=True,
                                    timeout=5,
                                    check=False
                                )
                                if not session_dir.exists():
                                    break
                            except:
                                pass
                            
                            # Fallback to shutil
                            shutil.rmtree(session_dir, ignore_errors=False)
                            break
                        except PermissionError as e:
                            if attempt < max_retries - 1:
                                time.sleep(1)
                            else:
                                # Mark for deletion on restart
                                try:
                                    marker_file = session_dir / '.delete_on_restart'
                                    marker_file.write_text('Delete this session directory')
                                except:
                                    pass
                                print(f"⚠️  Could not delete session directory (files in use)")
                                print(f"   Directory will be cleaned up on next restart: {session_dir}")
                        except Exception as e:
                            if attempt < max_retries - 1:
                                time.sleep(1)
                            else:
                                print(f"⚠️  Error deleting session: {e}")
                
            except Exception as e:
                print(f"⚠️  Error during session deletion: {e}")
                # Still try to remove from memory
                if session_id in self.sessions:
                    del self.sessions[session_id]
    
    def list_sessions(self) -> List[str]:
        """List all session IDs"""
        # Return list of session IDs sorted by update time
        sessions = sorted(
            self.sessions.values(),
            key=lambda s: s.metadata.get('updated_at', s.metadata.get('created_at', '')),
            reverse=True
        )
        
        return [s.session_id for s in sessions][:CONFIG['max_sessions']]
