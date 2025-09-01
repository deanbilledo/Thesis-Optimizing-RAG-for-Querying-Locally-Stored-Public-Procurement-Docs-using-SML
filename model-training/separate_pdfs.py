import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not found. Install with: pip install PyPDF2")
    exit(1)

try:
    import pytesseract
    from PIL import Image
    import pdf2image
except ImportError:
    print("OCR dependencies not found. Install with: pip install pytesseract pillow pdf2image")
    print("Also ensure tesseract is installed on your system:")
    print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    print("- macOS: brew install tesseract")
    print("- Ubuntu/Debian: sudo apt install tesseract-ocr")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, input_folder: str = "my_pdfs", output_file: str = "pdfs_dataset.jsonl"):
        self.input_folder = Path(input_folder)
        self.output_file = Path(output_file)
        self.processed_count = 0
        self.failed_count = 0
        
    def extract_text_pypdf(self, pdf_path: Path) -> Optional[str]:
        """Extract text using PyPDF2 for text-based PDFs."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1} of {pdf_path.name}: {e}")
                        continue
                
                # Check if we got meaningful text (not just whitespace/special chars)
                meaningful_text = ''.join(c for c in text if c.isalnum() or c.isspace())
                if len(meaningful_text.strip()) > 50:  # Threshold for meaningful text
                    return text.strip()
                else:
                    logger.info(f"Insufficient text extracted from {pdf_path.name}, will try OCR")
                    return None
                    
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed for {pdf_path.name}: {e}")
            return None
    
    def extract_text_ocr(self, pdf_path: Path) -> Optional[str]:
        """Extract text using OCR for scanned PDFs."""
        try:
            logger.info(f"Starting OCR for {pdf_path.name}")
            
            # Convert PDF to images
            pages = pdf2image.convert_from_path(pdf_path, dpi=300)
            text = ""
            
            for i, page in enumerate(pages):
                try:
                    # Use pytesseract to extract text
                    page_text = pytesseract.image_to_string(page, lang='eng')
                    if page_text.strip():
                        text += f"Page {i + 1}:\n{page_text}\n\n"
                except Exception as e:
                    logger.warning(f"OCR failed for page {i + 1} of {pdf_path.name}: {e}")
                    continue
            
            if text.strip():
                logger.info(f"OCR completed for {pdf_path.name}")
                return text.strip()
            else:
                logger.warning(f"OCR produced no text for {pdf_path.name}")
                return None
                
        except Exception as e:
            logger.error(f"OCR processing failed for {pdf_path.name}: {e}")
            return None
    
    def process_single_pdf(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single PDF file and return JSON record."""
        logger.info(f"Processing: {pdf_path.name}")
        
        # First try text extraction
        text = self.extract_text_pypdf(pdf_path)
        extraction_method = "text_extraction"
        
        # If text extraction failed or produced insufficient text, try OCR
        if not text:
            logger.info(f"Falling back to OCR for {pdf_path.name}")
            text = self.extract_text_ocr(pdf_path)
            extraction_method = "ocr"
        
        if not text:
            logger.error(f"Failed to extract any text from {pdf_path.name}")
            return None
        
        # Create JSON record
        record = {
            "filename": pdf_path.name,
            "filepath": str(pdf_path),
            "text": text,
            "extraction_method": extraction_method,
            "character_count": len(text),
            "word_count": len(text.split()),
            "file_size_bytes": pdf_path.stat().st_size
        }
        
        return record
    
    def process_all_pdfs(self):
        """Process all PDFs in the input folder."""
        if not self.input_folder.exists():
            logger.error(f"Input folder '{self.input_folder}' does not exist!")
            return
        
        pdf_files = list(self.input_folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in '{self.input_folder}'")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process each PDF and write to JSONL
        with open(self.output_file, 'w', encoding='utf-8') as outfile:
            for pdf_path in pdf_files:
                try:
                    record = self.process_single_pdf(pdf_path)
                    
                    if record:
                        # Write JSON record to file
                        json.dump(record, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        self.processed_count += 1
                        logger.info(f"✓ Successfully processed {pdf_path.name}")
                    else:
                        self.failed_count += 1
                        logger.error(f"✗ Failed to process {pdf_path.name}")
                        
                except Exception as e:
                    self.failed_count += 1
                    logger.error(f"✗ Error processing {pdf_path.name}: {e}")
                    continue
        
        # Print summary
        logger.info(f"\n{'='*50}")
        logger.info(f"PROCESSING COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Total PDFs found: {len(pdf_files)}")
        logger.info(f"Successfully processed: {self.processed_count}")
        logger.info(f"Failed to process: {self.failed_count}")
        logger.info(f"Output file: {self.output_file}")
        logger.info(f"{'='*50}")

def main():
    """Main function to run the PDF processor."""
    
    # You can customize these parameters
    INPUT_FOLDER = "my_pdfs"  # Change this if your folder has a different name
    OUTPUT_FILE = "pdfs_dataset.jsonl"  # Change this if you want a different output filename
    
    processor = PDFProcessor(INPUT_FOLDER, OUTPUT_FILE)
    processor.process_all_pdfs()

if __name__ == "__main__":
    main()