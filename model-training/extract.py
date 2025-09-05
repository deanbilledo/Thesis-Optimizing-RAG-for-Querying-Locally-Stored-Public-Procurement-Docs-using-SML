import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import json
import os
import re
import unicodedata
from pathlib import Path
import logging
from typing import List, Dict, Any
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFTextExtractor:
    def __init__(self, pdf_data_folder: str, output_file: str):
        self.pdf_data_folder = Path(pdf_data_folder)
        self.output_file = output_file
        
        # Configure Tesseract path (adjust if needed)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text or not text.strip():
            return ""
        
        # Remove common headers/footers patterns
        header_footer_patterns = [
            r'Page \d+ of \d+',
            r'Page \d+',
            r'\d+/\d+/\d+',  # dates
            r'^\d+$',  # standalone numbers
            r'Republic of the Philippines.*?(?=\n)',
            r'Department of.*?(?=\n)',
            r'PROCUREMENT.*?PLAN.*?(?=\n)',
        ]
        
        for pattern in header_footer_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Normalize currency symbols and special characters
        currency_replacements = {
            '₱': 'PHP',
            '$': 'USD',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '₩': 'KRW',
            ''': "'",
            ''': "'",
            '"': '"',
            '"': '"',
            '–': '-',
            '—': '-',
            '…': '...',
        }
        
        for symbol, replacement in currency_replacements.items():
            text = text.replace(symbol, replacement)
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\-.,;:!?()[\]{}/"\']+', ' ', text)
        
        # Final cleanup
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[str]:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            pages_text = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # First try to extract text directly
                text = page.get_text()
                
                # If no text or very little text, treat as scanned page
                if len(text.strip()) < 50:
                    logger.info(f"Page {page_num + 1} appears to be scanned, using OCR")
                    text = self.extract_text_with_ocr(page)
                
                cleaned_text = self.clean_text(text)
                if cleaned_text:
                    pages_text.append(cleaned_text)
            
            doc.close()
            return pages_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return []
    
    def extract_text_with_ocr(self, page) -> str:
        """Extract text from scanned page using Tesseract OCR"""
        try:
            # Get page as image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Use Tesseract to extract text
            text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')
            
            return text
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return ""
    
    def chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within last 200 characters
                sentence_end = text.rfind('.', end - 200, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def process_all_pdfs(self):
        """Process all PDFs in the pdf-data folder"""
        pdf_files = list(self.pdf_data_folder.glob("*.pdf"))
        
        if not pdf_files:
            logger.error(f"No PDF files found in {self.pdf_data_folder}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_chunks = []
        
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            
            try:
                pages_text = self.extract_text_from_pdf(pdf_file)
                
                for page_num, page_text in enumerate(pages_text, 1):
                    # Split page text into chunks
                    chunks = self.chunk_text(page_text)
                    
                    for chunk_num, chunk in enumerate(chunks, 1):
                        chunk_data = {
                            "file_name": pdf_file.name,
                            "page_number": page_num,
                            "chunk_number": chunk_num,
                            "text": chunk,
                            "metadata": {
                                "source": str(pdf_file),
                                "total_pages": len(pages_text),
                                "chunk_size": len(chunk)
                            }
                        }
                        all_chunks.append(chunk_data)
                
                logger.info(f"Extracted {len(pages_text)} pages from {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                continue
        
        # Save to JSONL file
        self.save_to_jsonl(all_chunks)
        logger.info(f"Processing complete. Total chunks: {len(all_chunks)}")
    
    def save_to_jsonl(self, chunks: List[Dict[str, Any]]):
        """Save chunks to JSONL file"""
        try:
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    json.dump(chunk, f, ensure_ascii=False)
                    f.write('\n')
            
            logger.info(f"Saved {len(chunks)} chunks to {self.output_file}")
            
        except Exception as e:
            logger.error(f"Error saving to JSONL: {str(e)}")

def main():
    # Set up paths
    current_dir = Path(__file__).parent
    pdf_data_folder = current_dir / "pdf-data"
    output_file = current_dir / "raw-chunks.jsonl"
    
    # Check if pdf-data folder exists
    if not pdf_data_folder.exists():
        logger.error(f"PDF data folder not found: {pdf_data_folder}")
        return
    
    # Initialize extractor and process files
    extractor = PDFTextExtractor(pdf_data_folder, output_file)
    extractor.process_all_pdfs()

if __name__ == "__main__":
    main()