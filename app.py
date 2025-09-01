# app.py
from flask import Flask, request, jsonify, send_from_directory
import os
import json
import tempfile
import faiss
import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
import requests
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = Flask(__name__, static_folder='.', static_url_path='')

# Configuration
UPLOAD_FOLDER = 'uploads'
DB_FOLDER = 'db'
VECTOR_DIMENSION = 384  # Dimension for embeddings (depends on model used)
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks

# Fine-tuned model configuration
FINETUNED_MODEL_PATH = "model-training/tinyllama_procurement_20250830_104738"  # Your actual model path
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# Create necessary directories
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(DB_FOLDER).mkdir(exist_ok=True)

# Initialize embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Small, fast model

# Initialize fine-tuned model (lazy loading)
finetuned_model = None
finetuned_tokenizer = None

# Initialize FAISS index
faiss_index = faiss.IndexFlatL2(VECTOR_DIMENSION)  # Renamed from 'index' to 'faiss_index'
document_chunks = []  # Store text chunks corresponding to vectors

def load_finetuned_model():
    """Load the fine-tuned TinyLlama model."""
    global finetuned_model, finetuned_tokenizer
    
    if finetuned_model is None:
        try:
            print("Loading fine-tuned TinyLlama model...")
            
            # Check device availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Load tokenizer
            finetuned_tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_NAME,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # Add pad token if it doesn't exist
            if finetuned_tokenizer.pad_token is None:
                finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
                finetuned_tokenizer.pad_token_id = finetuned_tokenizer.eos_token_id
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            if device == "cpu":
                base_model = base_model.to(device)
            
            # Load LoRA weights
            finetuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
            finetuned_model.eval()
            
            print("Fine-tuned model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            print("Falling back to base model...")
            
            # Fallback to base model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            finetuned_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            if finetuned_tokenizer.pad_token is None:
                finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
            
            finetuned_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            if device == "cpu":
                finetuned_model = finetuned_model.to(device)
    
    return finetuned_model, finetuned_tokenizer

# Load existing index if available
def load_index():
    global faiss_index, document_chunks  # Change index to faiss_index
    if os.path.exists(f"{DB_FOLDER}/index.faiss") and os.path.exists(f"{DB_FOLDER}/chunks.json"):
        try:
            faiss_index = faiss.read_index(f"{DB_FOLDER}/index.faiss")  # Change index to faiss_index
            with open(f"{DB_FOLDER}/chunks.json", 'r') as f:
                document_chunks = json.load(f)
            print(f"Loaded existing index with {len(document_chunks)} chunks")
        except Exception as e:
            print(f"Error loading index: {e}")
            # Initialize new index
            faiss_index = faiss.IndexFlatL2(VECTOR_DIMENSION)  # Change index to faiss_index
            document_chunks = []

# Save index
def save_index():
    faiss.write_index(faiss_index, f"{DB_FOLDER}/index.faiss")  # Change index to faiss_index
    with open(f"{DB_FOLDER}/chunks.json", 'w') as f:
        json.dump(document_chunks, f)
    print(f"Index saved with {len(document_chunks)} chunks")

# Text chunking function
def chunk_text(text, filename="", page_num=0):
    chunks = []
    i = 0
    while i < len(text):
        # Get chunk with overlap
        chunk = text[i:i + CHUNK_SIZE]
        if chunk:
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "page": page_num,
                    "start_char": i,
                    "end_char": min(i + CHUNK_SIZE, len(text))
                }
            })
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# Extract text from PDF
def extract_pdf_text(file_path):
    chunks = []
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    page_chunks = chunk_text(text, os.path.basename(file_path), i)
                    chunks.extend(page_chunks)
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
    return chunks

# Add a document to the index
def add_document_to_index(file_path):
    chunks = extract_pdf_text(file_path)
    if not chunks:
        return {"success": False, "message": "No text extracted from document"}
    
    # Get embeddings for all chunks
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    
    # Add to FAISS index
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    
    # Store chunk information
    start_idx = len(document_chunks)
    for i, chunk in enumerate(chunks):
        document_chunks.append(chunk)
    
    save_index()
    return {"success": True, "chunks_added": len(chunks)}

# Query using fine-tuned TinyLlama with retrieval augmentation
def query_finetuned_model(query, top_k=3):
    # Get query embedding
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search in FAISS
    D, I = faiss_index.search(query_embedding, top_k)
    
    if len(I[0]) == 0:
        return {"response": "No relevant information found. Please upload some documents first."}
    
    # Get relevant contexts
    contexts = []
    for idx in I[0]:
        if idx < len(document_chunks):
            contexts.append(document_chunks[idx]["text"])
    
    # Build prompt with context
    context_text = "\n\n".join(contexts)
    prompt = f"""Document Context: {context_text}

Question: {query}

Answer: """
    
    # Load model if not already loaded
    model_instance, tokenizer = load_finetuned_model()
    
    if model_instance is None:
        return {"error": "Failed to load fine-tuned model"}
    
    try:
        # Load model if not already loaded
        model_instance, tokenizer = load_finetuned_model()
        
        if model_instance is None:
            return {"error": "Failed to load fine-tuned model"}
        
        # Get device from model
        device = next(model_instance.parameters()).device
        
        # Tokenize input and move to correct device
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = inputs.to(device)
        
        # Create attention mask
        attention_mask = torch.ones_like(inputs).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model_instance.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=inputs.shape[1] + 200,  # Generate up to 200 new tokens
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        generated_text = response[len(prompt):].strip()
        
        return {"response": generated_text if generated_text else "I couldn't generate a response based on the provided context."}
        
    except Exception as e:
        return {"error": f"Error generating response: {str(e)}"}

# Legacy Ollama function (keeping as fallback)
def query_ollama(query, top_k=3):
    # Get query embedding
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search in FAISS
    D, I = faiss_index.search(query_embedding, top_k)
    
    if len(I[0]) == 0:
        return {"response": "No relevant information found. Please upload some documents first."}
    
    # Get relevant contexts
    contexts = []
    for idx in I[0]:
        if idx < len(document_chunks):
            contexts.append(document_chunks[idx]["text"])
    
    # Build prompt with context
    context_text = "\n\n".join(contexts)
    prompt = f"""
    You are an expert in procurement documents. 
    Use the following information to answer the query.
    
    Context information:
    {context_text}
    
    Query: {query}
    
    Answer based only on the provided context. If the information is not in the context, say that you don't know.
    """
    
    # Query Ollama
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:3b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_gpu": 0  # Force CPU mode
                }
            }
        )
        
        if response.status_code == 200:
            return {"response": response.json()["response"]}
        else:
            return {"error": f"Ollama error: {response.text}"}
    except Exception as e:
        return {"error": f"Error querying Ollama: {str(e)}"}

# Flask routes
@app.route('/')
def serve_index():  # Renamed from 'index' to 'serve_index'
    return send_from_directory('.', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400
    
    # Save file temporarily
    temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_path)
    
    # Process file
    result = add_document_to_index(temp_path)
    
    return jsonify(result)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    # Use fine-tuned model by default, fallback to Ollama if specified
    use_ollama = data.get('use_ollama', False)
    
    if use_ollama:
        result = query_ollama(data['query'])
    else:
        result = query_finetuned_model(data['query'])
    
    return jsonify(result)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/models', methods=['GET'])
def get_available_models():
    """Get information about available models."""
    return jsonify({
        "models": [
            {
                "name": "finetuned_tinyllama",
                "display_name": "Fine-tuned TinyLlama (Procurement)",
                "description": "TinyLlama model fine-tuned on procurement documents",
                "available": os.path.exists(FINETUNED_MODEL_PATH),
                "default": True
            },
            {
                "name": "ollama_llama3.2",
                "display_name": "Llama 3.2 3B (Ollama)",
                "description": "General-purpose Llama 3.2 model via Ollama",
                "available": check_ollama_connection(),
                "default": False
            }
        ]
    })

@app.route('/status', methods=['GET'])
def status():
    # Check if fine-tuned model exists
    finetuned_available = os.path.exists(FINETUNED_MODEL_PATH)
    
    return jsonify({
        "documents_count": len(set(chunk["metadata"]["source"] for chunk in document_chunks)),
        "chunks_count": len(document_chunks),
        "ollama_status": "connected" if check_ollama_connection() else "disconnected",
        "finetuned_model_available": finetuned_available,
        "finetuned_model_path": FINETUNED_MODEL_PATH,
        "model_loaded": finetuned_model is not None
    })

def check_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

# Initialize index on startup
load_index()

if __name__ == '__main__':
    app.run(debug=True, port=5000)