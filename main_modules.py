"""
================================================================================
MAIN MODULES - CORE COMPONENTS OF THE RAG SYSTEM
================================================================================
Optimizing Retrieval-Augmented Generation for Querying Locally Stored
Public Procurement Documents using Small Language Models

This file contains the essential core modules extracted from the RAG system
for documentation and manuscript purposes.

Components:
    1. Configuration - System parameters and settings
    2. Document Processing - PDF text extraction and chunking
    3. Embedding Generation - Semantic vector representation
    4. Vector Database - ChromaDB storage and retrieval
    5. Language Model Integration - LLM loading with LoRA adapter
    6. Retrieval System - Hybrid semantic search
    7. Response Generation - RAG-based answer generation
    8. Session Management - Multi-session handling
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import json
import torch
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==============================================================================
# 1. CONFIGURATION MODULE
# ==============================================================================
# System-wide configuration parameters for the RAG pipeline

CONFIG = {
    # Document Processing Limits
    'max_pdfs_per_session': 6,          # Maximum PDFs allowed per session
    'max_pages_per_pdf': 15,            # Maximum pages to process per PDF
    'max_total_size_mb': 20,            # Maximum total upload size in MB
    'max_sessions': 10,                 # Maximum concurrent sessions
    
    # Text Chunking Parameters
    'chunk_size': 1200,                 # Characters per chunk
    'chunk_overlap': 50,                # Overlap between consecutive chunks
    
    # Retrieval Configuration
    'top_k_chunks': 3,                  # Number of chunks to retrieve
    
    # Model Configuration
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',  # Embedding model
    'base_model': 'google/gemma-3-1b-it',                         # Base LLM
    'lora_adapter_path': './model',                               # Fine-tuned LoRA adapter
    
    # Generation Parameters
    'max_new_tokens': 512,              # Maximum tokens to generate
    'temperature': 0.3,                 # Sampling temperature (0.0 = deterministic)
    'do_sample': True,                  # Enable sampling for diverse outputs
    
    # Retrieval Scoring Weights (Hybrid Retrieval)
    'retrieval_weights': {
        'cosine': 0.90,                 # Cosine similarity weight (primary)
        'llm_judge': 0.00,              # LLM judge score weight
        'structural': 0.05,             # Document structure weight
        'metadata': 0.00,               # Metadata relevance weight
        'mmr': 0.05                     # Maximal Marginal Relevance (diversity)
    },
    
    # Document Section Tags for Classification
    'section_tags': [
        'CONTRACT_AMOUNT',              # Budget/cost information
        'BIDDER_INFO',                  # Bidder/supplier details
        'LEGAL_CLAUSES',                # Legal terms and conditions
        'COMPLIANCE_REQUIREMENTS',       # Regulatory requirements
        'TIMELINE',                     # Dates and schedules
        'TECHNICAL_SPECS',              # Technical specifications
        'GENERAL',                      # General content
        'TABLE_DATA'                    # Tabular data
    ]
}


# ==============================================================================
# 2. GPU/DEVICE DETECTION
# ==============================================================================
def check_gpu() -> Dict:
    """
    Check GPU availability and return device information.
    
    Returns:
        Dict containing GPU availability status and specifications.
    """
    if torch.cuda.is_available():
        return {
            'available': True,
            'name': torch.cuda.get_device_name(0),
            'memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9,
            'memory_allocated': torch.cuda.memory_allocated(0) / 1e9,
        }
    return {'available': False}


# ==============================================================================
# 3. DOCUMENT SECTION TAGGING
# ==============================================================================
def tag_section(text: str) -> str:
    """
    Identify document section based on keyword matching.
    
    This lightweight tagging system categorizes text chunks into predefined
    sections to improve retrieval relevance for domain-specific queries.
    
    Args:
        text: The text content to classify.
        
    Returns:
        Section tag string (e.g., 'CONTRACT_AMOUNT', 'BIDDER_INFO', etc.)
    """
    text_lower = text.lower()
    
    # Keyword-based classification rules
    if any(kw in text_lower for kw in ['amount', 'price', 'cost', 'budget', 'php', '$']):
        return 'CONTRACT_AMOUNT'
    elif any(kw in text_lower for kw in ['bidder', 'supplier', 'vendor', 'contractor']):
        return 'BIDDER_INFO'
    elif any(kw in text_lower for kw in ['legal', 'clause', 'terms', 'conditions']):
        return 'LEGAL_CLAUSES'
    elif any(kw in text_lower for kw in ['compliance', 'requirement', 'regulation']):
        return 'COMPLIANCE_REQUIREMENTS'
    elif any(kw in text_lower for kw in ['timeline', 'schedule', 'deadline', 'date']):
        return 'TIMELINE'
    elif any(kw in text_lower for kw in ['technical', 'specification', 'specs']):
        return 'TECHNICAL_SPECS'
    elif '|' in text or text.count('\t') > 3:  # Table detection
        return 'TABLE_DATA'
    else:
        return 'GENERAL'


# ==============================================================================
# 4. PDF DOCUMENT PROCESSOR
# ==============================================================================
class PDFProcessor:

    @staticmethod
    def extract_text(pdf_path: str) -> List[Dict]:

        chunks = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Enforce page limit
            if total_pages > CONFIG['max_pages_per_pdf']:
                raise ValueError(
                    f"PDF has {total_pages} pages, "
                    f"max {CONFIG['max_pages_per_pdf']} allowed"
                )
            
            # Process each page
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                if text and text.strip() and len(text.strip()) > 50:
                    # Split page text into smaller chunks
                    chunk_texts = PDFProcessor._chunk_text(text)
                    
                    for chunk_text in chunk_texts:
                        if chunk_text.strip():
                            # Classify the chunk by section type
                            section_tag = tag_section(chunk_text)
                            
                            chunks.append({
                                'text': chunk_text,
                                'page': page_num + 1,
                                'source': os.path.basename(pdf_path),
                                'section_tag': section_tag
                            })
        
        return chunks
    
    @staticmethod
    def _chunk_text(text: str) -> List[str]:

        chunk_size = CONFIG['chunk_size']
        overlap = CONFIG['chunk_overlap']
        
        lines = text.split('\n')
        chunks = []
        current_chunk_lines = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 for newline character
            
            # If adding this line exceeds chunk size, finalize current chunk
            if current_size + line_size > chunk_size and current_chunk_lines:
                chunk_text = '\n'.join(current_chunk_lines).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                
                # Calculate overlap: keep last few lines for context continuity
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
            
            current_chunk_lines.append(line)
            current_size += line_size
        
        # Add final chunk
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines).strip()
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks


# ==============================================================================
# 5. RAG SESSION - CORE RAG PIPELINE
# ==============================================================================
class RAGSession:
    """
    RAG Session Manager - Core of the Retrieval-Augmented Generation Pipeline.
    
    This class encapsulates all RAG functionality:
    - Document ingestion and indexing
    - Embedding generation and storage
    - Semantic retrieval with hybrid scoring
    - LLM-based response generation
    - Chat history management
    
    Each session maintains isolated context with its own:
    - ChromaDB vector database
    - Document collection
    - Chat history
    """
    
    def __init__(self, session_id: str, session_name: str, session_dir: Path):
        """
        Initialize a new RAG session.
        
        Args:
            session_id: Unique identifier for the session.
            session_name: Human-readable session name.
            session_dir: Directory path for session data storage.
        """
        self.session_id = session_id
        self.session_name = session_name
        self.session_dir = session_dir
        self.pdf_dir = session_dir / "pdfs"
        self.db_dir = session_dir / "chroma_db"
        self.metadata_file = session_dir / "metadata.json"
        
        # Create required directories
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model references (lazy loading for efficiency)
        self._embedding_model = None
        self._llm_model = None
        self._llm_tokenizer = None
        self._chroma_client = None
        self._collection = None
        self._device = None
        
        # Load session metadata
        self.metadata = self._load_metadata()
        self.chat_history = self.metadata.get('chat_history', [])
        self.documents = self.metadata.get('documents', [])
    
    # --------------------------------------------------------------------------
    # METADATA MANAGEMENT
    # --------------------------------------------------------------------------
    def _load_metadata(self) -> Dict:
        """Load session metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'session_id': self.session_id,
            'session_name': self.session_name,
            'created_at': datetime.now().isoformat(),
            'documents': [],
            'chat_history': []
        }
    
    def _save_metadata(self):
        """Persist session metadata to disk."""
        self.metadata['documents'] = self.documents
        self.metadata['chat_history'] = self.chat_history
        self.metadata['updated_at'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    # --------------------------------------------------------------------------
    # MODEL LOADING - EMBEDDING MODEL
    # --------------------------------------------------------------------------
    @property
    def embedding_model(self):
        """
        Lazy load the sentence embedding model.
        
        Uses sentence-transformers/all-MiniLM-L6-v2 for generating
        384-dimensional dense vector representations of text.
        
        Returns:
            SentenceTransformer model instance.
        """
        if self._embedding_model is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Check for local model first (offline capability)
            local_model_path = Path('./embedding_model')
            if local_model_path.exists():
                model_path = str(local_model_path)
            else:
                model_path = CONFIG['embedding_model']
            
            self._embedding_model = SentenceTransformer(model_path, device=device)
        
        return self._embedding_model
    
    # --------------------------------------------------------------------------
    # MODEL LOADING - LANGUAGE MODEL WITH LoRA
    # --------------------------------------------------------------------------
    def load_llm(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._device = device
        
        # Check for local base model (offline capability)
        local_base_model = Path('./base_model')
        if local_base_model.exists():
            base_model_path = str(local_base_model)
        else:
            base_model_path = CONFIG['base_model']
        
        # Load tokenizer from LoRA adapter directory
        self._llm_tokenizer = AutoTokenizer.from_pretrained(
            CONFIG['lora_adapter_path']
        )
        
        # Load base model with appropriate precision
        if device == 'cuda':
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,      # Half precision for GPU
                device_map='auto',               # Automatic device placement
                low_cpu_mem_usage=True
            )
        else:
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float32,      # Full precision for CPU
                low_cpu_mem_usage=True
            )
        
        # Load and merge LoRA adapter
        self._llm_model = PeftModel.from_pretrained(
            self._llm_model,
            CONFIG['lora_adapter_path']
        )
        
        # Move to device and set to evaluation mode
        if device == 'cpu':
            self._llm_model = self._llm_model.to(device)
        self._llm_model.eval()
    
    # --------------------------------------------------------------------------
    # VECTOR DATABASE - CHROMADB
    # --------------------------------------------------------------------------
    @property
    def chroma_client(self):
        """
        Lazy load ChromaDB persistent client.
        
        ChromaDB is used as the vector database for storing and retrieving
        document embeddings with metadata.
        """
        if self._chroma_client is None:
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
        return self._chroma_client
    
    @property
    def collection(self):
        """Get or create the ChromaDB collection for this session."""
        if self._collection is None:
            _ = self.chroma_client  # Initialize client
        return self._collection
    
    # --------------------------------------------------------------------------
    # DOCUMENT INGESTION
    # --------------------------------------------------------------------------
    def add_documents(self, uploaded_files) -> List[Dict]:

        results = []
        
        # Check for duplicates
        existing = {doc['filename'] for doc in self.documents}
        new_files = [f for f in uploaded_files if f.name not in existing]
        
        # Check document limit
        if len(self.documents) + len(new_files) > CONFIG['max_pdfs_per_session']:
            return [{'success': False, 'error': 'Document limit exceeded'}]
        
        for uploaded_file in new_files:
            try:
                # Save PDF to session directory
                pdf_path = self.pdf_dir / uploaded_file.name
                with open(pdf_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract text chunks with section tagging
                chunks = PDFProcessor.extract_text(str(pdf_path))
                
                # Generate embeddings for all chunks
                texts = [chunk['text'] for chunk in chunks]
                embeddings = self.embedding_model.encode(
                    texts,
                    convert_to_numpy=True
                ).tolist()
                
                # Prepare metadata for ChromaDB
                ids = [f"{uploaded_file.name}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [{
                    'source': chunk['source'],
                    'page': chunk['page'],
                    'section_tag': chunk.get('section_tag', 'GENERAL')
                } for chunk in chunks]
                
                # Add to vector database
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas
                )
                
                # Update session metadata
                self.documents.append({
                    'filename': uploaded_file.name,
                    'chunks': len(chunks),
                    'pages': max(c['page'] for c in chunks),
                    'added_at': datetime.now().isoformat()
                })
                
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
    
    # --------------------------------------------------------------------------
    # SEMANTIC RETRIEVAL
    # --------------------------------------------------------------------------
    def retrieve_context(self, query: str, top_k: int = None) -> List[Dict]:

        if top_k is None:
            top_k = CONFIG['top_k_chunks']
        
        # Encode query to embedding vector
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()
        
        # Query ChromaDB for similar chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 3  # Get more candidates for reranking
        )
        
        chunks = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                # Calculate cosine similarity (1 - distance)
                distance = results['distances'][0][i]
                cosine_score = 1 - distance
                
                # Apply retrieval weights
                weights = CONFIG['retrieval_weights']
                final_score = weights['cosine'] * cosine_score
                
                chunks.append({
                    'content': doc,
                    'score': final_score,
                    'cosine_score': cosine_score,
                    'source': results['metadatas'][0][i]['source'],
                    'page': results['metadatas'][0][i]['page'],
                    'section_tag': results['metadatas'][0][i].get('section_tag', 'GENERAL')
                })
        
        # Sort by score and return top-k
        return sorted(chunks, key=lambda x: x['score'], reverse=True)[:top_k]
    
    # --------------------------------------------------------------------------
    # RESPONSE GENERATION
    # --------------------------------------------------------------------------
    def generate_response(self, question: str) -> Tuple[str, Dict]:

        start_time = time.time()
        
        # Load LLM if not already loaded
        if self._llm_model is None:
            self.load_llm()
        
        # Retrieve relevant context chunks
        retrieval_start = time.time()
        chunks = self.retrieve_context(question)
        retrieval_time = time.time() - retrieval_start
        
        if not chunks:
            return "No relevant information found in the documents.", {}
        
        # Build context from retrieved chunks
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"[{chunk['source']}, Page {chunk['page']}]\n{chunk['content']}"
            )
        context = "\n\n".join(context_parts)
        
        # Construct prompt using chat template
        messages = [
            {"role": "user", "content": f"""Answer based on the document content below.

            DOCUMENT CONTENT:
            {context}

            Question: {question}

            Answer:"""}
                    ]
        
        # Apply chat template and tokenize
        formatted_prompt = self._llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self._llm_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        # Generate response
        generation_start = time.time()
        with torch.no_grad():
            outputs = self._llm_model.generate(
                **inputs,
                max_new_tokens=CONFIG['max_new_tokens'],
                temperature=CONFIG['temperature'],
                do_sample=CONFIG['do_sample'],
                pad_token_id=self._llm_tokenizer.eos_token_id
            )
        generation_time = time.time() - generation_start
        
        # Decode and extract response
        full_response = self._llm_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Remove input prompt from response
        input_text = self._llm_tokenizer.decode(
            inputs['input_ids'][0],
            skip_special_tokens=True
        )
        if full_response.startswith(input_text):
            response = full_response[len(input_text):].strip()
        else:
            response = full_response
        
        # Compile debug information
        debug_info = {
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': time.time() - start_time,
            'chunks_retrieved': len(chunks),
            'input_tokens': len(inputs['input_ids'][0]),
            'output_tokens': len(outputs[0]) - len(inputs['input_ids'][0])
        }
        
        return response, debug_info
    
    # --------------------------------------------------------------------------
    # CHAT HISTORY MANAGEMENT
    # --------------------------------------------------------------------------
    def add_message(self, role: str, content: str):
        """Add a message to chat history."""
        self.chat_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        self._save_metadata()
    
    def clear_chat_history(self):
        """Clear all chat history."""
        self.chat_history = []
        self._save_metadata()


# ==============================================================================
# 6. SESSION MANAGER
# ==============================================================================
class SessionManager:
    """
    Multi-Session Management System.
    
    Handles creation, retrieval, and deletion of RAG sessions.
    Each session maintains isolated document collections and chat histories.
    """
    
    def __init__(self, sessions_dir: str = "./sessions"):
        """
        Initialize the session manager.
        
        Args:
            sessions_dir: Base directory for storing session data.
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(exist_ok=True)
        self.sessions: Dict[str, RAGSession] = {}
        self._load_sessions()
    
    def _load_sessions(self):
        """Load existing sessions from disk on startup."""
        for session_dir in self.sessions_dir.iterdir():
            if session_dir.is_dir():
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
                        print(f"Warning: Could not load session: {e}")
    
    def create_session(self, session_name: str) -> str:
        """
        Create a new RAG session.
        
        Args:
            session_name: Human-readable name for the session.
            
        Returns:
            Unique session ID string.
        """
        # Enforce session limit
        if len(self.sessions) >= CONFIG['max_sessions']:
            oldest = min(
                self.sessions.values(),
                key=lambda s: s.metadata.get('updated_at', '')
            )
            self.delete_session(oldest.session_id)
        
        # Generate unique session ID
        session_id = hashlib.md5(
            f"{session_name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Create and store session
        session_dir = self.sessions_dir / session_id
        session = RAGSession(session_id, session_name, session_dir)
        self.sessions[session_id] = session
        session._save_metadata()
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[RAGSession]:
        """Retrieve a session by ID."""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str):
        """Delete a session and clean up resources."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session_dir = session.session_dir
            del self.sessions[session_id]
            
            # Remove session directory
            import shutil
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)
    
    def list_sessions(self) -> List[str]:
        """List all session IDs sorted by last update time."""
        return sorted(
            self.sessions.keys(),
            key=lambda sid: self.sessions[sid].metadata.get('updated_at', ''),
            reverse=True
        )


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================
if __name__ == "__main__":
    """
    Example usage of the RAG system components.
    """
    print("=" * 60)
    print("RAG System - Core Modules Demonstration")
    print("=" * 60)
    
    # Check GPU availability
    gpu_info = check_gpu()
    print(f"\nGPU Available: {gpu_info['available']}")
    if gpu_info['available']:
        print(f"GPU Name: {gpu_info['name']}")
        print(f"GPU Memory: {gpu_info['memory_total']:.2f} GB")
    
    # Initialize session manager
    print("\nInitializing Session Manager...")
    manager = SessionManager()
    
    # Create a new session
    session_id = manager.create_session("Demo Session")
    session = manager.get_session(session_id)
    print(f"Created session: {session_id}")
    
    # Display configuration
    print("\nConfiguration:")
    print(f"  Chunk Size: {CONFIG['chunk_size']} characters")
    print(f"  Top-K Retrieval: {CONFIG['top_k_chunks']} chunks")
    print(f"  Base Model: {CONFIG['base_model']}")
    print(f"  Embedding Model: {CONFIG['embedding_model']}")
    
    print("\n" + "=" * 60)
    print("Core modules loaded successfully!")
    print("=" * 60)
