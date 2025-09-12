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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

app = Flask(__name__, static_folder='.', static_url_path='')

# Configuration
UPLOAD_FOLDER = 'uploads'
DB_FOLDER = 'db'
VECTOR_DIMENSION = 384  # Dimension for embeddings (depends on model used)
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks

# Available models configuration
AVAILABLE_MODELS = {
    "gemma_procurement_lora": {
        "name": "Gemma 2B (LoRA Fine-tuned)",
        "type": "lora",
        "base_model": "model-training/gemma-2b",
        "lora_path": "model-training/procurement-lora-checkpoint-gemma",
        "description": "Gemma 2B fine-tuned on procurement documents using LoRA"
    },
    "gemma_procurement_merged": {
        "name": "Gemma 2B (Merged Fine-tuned)",
        "type": "merged",
        "model_path": "model-training/procurement-merged-gemma",
        "description": "Gemma 2B with LoRA weights merged - ready to use"
    },
    "phi3_procurement_lora": {
        "name": "Phi-3 Mini (LoRA Fine-tuned)",
        "type": "lora",
        "base_model": "model-training/Phi-3-mini-4k-instruct",
        "lora_path": "model-training/procurement-lora-checkpoint-phi",
        "description": "Phi-3 Mini fine-tuned on procurement documents using LoRA"
    },
    "phi3_procurement_merged": {
        "name": "Phi-3 Mini (Merged Fine-tuned)",
        "type": "merged",
        "model_path": "model-training/procurement-merged-phi",
        "description": "Phi-3 Mini with LoRA weights merged"
    },
    "llama_procurement_lora": {
        "name": "Llama 3.2 3B (LoRA Fine-tuned)",
        "type": "lora",
        "base_model": "model-training/Llama-3.2-3B-Instruct",
        "lora_path": "model-training/procurement-lora-checkpoint-llama",
        "description": "Llama 3.2 3B fine-tuned on procurement documents using LoRA"
    },
    "llama_procurement_merged": {
        "name": "Llama 3.2 3B (Merged Fine-tuned)",
        "type": "merged",
        "model_path": "model-training/procurement-merged-llama",
        "description": "Llama 3.2 3B with LoRA weights merged"
    },
    "gemma_base": {
        "name": "Gemma 2B (Base Model)",
        "type": "base",
        "model_path": "gemma-2b",
        "description": "Base Gemma 2B model without fine-tuning"
    }
}

# Default model
DEFAULT_MODEL = "gemma_procurement_merged"

# Create necessary directories
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(DB_FOLDER).mkdir(exist_ok=True)

# Initialize embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Small, fast model

# Global model variables
current_model = None
current_tokenizer = None
current_model_id = None

# Initialize FAISS index
faiss_index = faiss.IndexFlatL2(VECTOR_DIMENSION)  # Renamed from 'index' to 'faiss_index'
document_chunks = []  # Store text chunks corresponding to vectors

def get_available_models():
    """Get list of available models with their status."""
    models = []
    
    for model_id, config in AVAILABLE_MODELS.items():
        model_info = {
            "id": model_id,
            "name": config["name"],
            "type": config["type"],
            "description": config["description"],
            "available": False,
            "path": ""
        }
        
        # Check availability based on model type
        if config["type"] == "lora":
            # Check if both base model and LoRA path exist
            base_available = check_model_exists(config["base_model"])
            lora_available = os.path.exists(config["lora_path"])
            model_info["available"] = base_available and lora_available
            model_info["path"] = config["lora_path"]
            model_info["base_model"] = config["base_model"]
            
        elif config["type"] == "merged":
            # Check if merged model path exists
            model_info["available"] = os.path.exists(config["model_path"])
            model_info["path"] = config["model_path"]
            
        elif config["type"] == "base":
            # Check if base model is available
            model_info["available"] = check_model_exists(config["model_path"])
            model_info["path"] = config["model_path"]
        
        models.append(model_info)
    
    return models

def check_model_exists(model_path):
    """Check if a model exists locally or on HuggingFace."""
    if os.path.exists(model_path):
        return True
    
    # For HuggingFace models, we'll assume they're available
    # You could add a more sophisticated check here
    if not model_path.startswith('./') and not model_path.startswith('/'):
        return True
    
    return False

def load_model(model_id):
    """Load a specific model by ID."""
    global current_model, current_tokenizer, current_model_id
    
    if model_id == current_model_id and current_model is not None:
        print(f"Model {model_id} already loaded")
        return current_model, current_tokenizer
    
    # Clear previous model from memory
    if current_model is not None:
        del current_model
        del current_tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model ID: {model_id}")
    
    config = AVAILABLE_MODELS[model_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {config['name']} on {device}")
    
    try:
        # Setup quantization for GPU
        quantization_config = None
        if device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        
        if config["type"] == "lora":
            # Load base model + LoRA
            print(f"Loading base model: {config['base_model']}")
            
            # Load tokenizer
            current_tokenizer = AutoTokenizer.from_pretrained(
                config["base_model"],
                trust_remote_code=True,
                padding_side="right"
            )
            
            # Add pad token if missing
            if current_tokenizer.pad_token is None:
                current_tokenizer.pad_token = current_tokenizer.eos_token
                current_tokenizer.pad_token_id = current_tokenizer.eos_token_id
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                config["base_model"],
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            if device == "cpu":
                base_model = base_model.to(device)
            
            # Load LoRA weights
            print(f"Loading LoRA weights from: {config['lora_path']}")
            current_model = PeftModel.from_pretrained(base_model, config["lora_path"])
            current_model.eval()
            
        elif config["type"] == "merged":
            # Load merged model
            print(f"Loading merged model from: {config['model_path']}")
            
            current_tokenizer = AutoTokenizer.from_pretrained(
                config["model_path"],
                trust_remote_code=True,
                padding_side="right"
            )
            
            if current_tokenizer.pad_token is None:
                current_tokenizer.pad_token = current_tokenizer.eos_token
                current_tokenizer.pad_token_id = current_tokenizer.eos_token_id
            
            current_model = AutoModelForCausalLM.from_pretrained(
                config["model_path"],
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            if device == "cpu":
                current_model = current_model.to(device)
            
        elif config["type"] == "base":
            # Load base model
            print(f"Loading base model: {config['model_path']}")
            
            current_tokenizer = AutoTokenizer.from_pretrained(
                config["model_path"],
                trust_remote_code=True,
                padding_side="right"
            )
            
            if current_tokenizer.pad_token is None:
                current_tokenizer.pad_token = current_tokenizer.eos_token
                current_tokenizer.pad_token_id = current_tokenizer.eos_token_id
            
            current_model = AutoModelForCausalLM.from_pretrained(
                config["model_path"],
                quantization_config=quantization_config,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            if device == "cpu":
                current_model = current_model.to(device)
        
        current_model_id = model_id
        print(f"Successfully loaded model: {config['name']}")
        return current_model, current_tokenizer
        
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        raise e

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
    embeddings = embedding_model.encode(texts)
    
    # Add to FAISS index
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    
    # Store chunk information
    start_idx = len(document_chunks)
    for i, chunk in enumerate(chunks):
        document_chunks.append(chunk)
    
    save_index()
    return {"success": True, "chunks_added": len(chunks)}

# Query using selected model with retrieval augmentation
def query_model(query, model_id=None, top_k=3):
    if model_id is None:
        model_id = DEFAULT_MODEL
    
    # Get query embedding
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    
    # Search in FAISS
    D, I = faiss_index.search(query_embedding, top_k)
    
    if len(I[0]) == 0:
        return {"response": "No relevant information found. Please upload some documents first."}
    
    # Get relevant contexts
    contexts = []
    sources = []
    for idx in I[0]:
        if idx < len(document_chunks):
            contexts.append(document_chunks[idx]["text"])
            sources.append(document_chunks[idx]["metadata"]["source"])
    
    # Build prompt with context
    context_text = "\n\n".join(contexts)
    
    # Use procurement-specific prompt format
    prompt = f"""### Instruction:
Answer the following question based on the provided procurement document context. Be specific and accurate.

### Context:
{context_text}

### Question:
{query}

### Answer:
"""
    
    try:
        # Load the specified model
        model_instance, tokenizer = load_model(model_id)
        
        # Get device from model
        device = next(model_instance.parameters()).device
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = inputs.to(device)
        
        # Create attention mask
        attention_mask = torch.ones_like(inputs).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model_instance.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=inputs.shape[1] + 150,  # Generate up to 150 new tokens
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        generated_text = response[len(prompt):].strip()
        
        return {
            "response": generated_text if generated_text else "I couldn't generate a response based on the provided context.",
            "sources": list(set(sources)),
            "model_used": AVAILABLE_MODELS[model_id]["name"]
        }
        
    except Exception as e:
        return {"error": f"Error generating response: {str(e)}"}

# Legacy Ollama function (keeping as fallback)
def query_ollama(query, top_k=3):
    # Get query embedding
    query_embedding = embedding_model.encode([query])
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
    
    # Get selected model ID, default to DEFAULT_MODEL
    model_id = data.get('model_id', DEFAULT_MODEL)
    use_ollama = data.get('use_ollama', False)
    
    if use_ollama:
        result = query_ollama(data['query'])
    else:
        result = query_model(data['query'], model_id)
    
    return jsonify(result)

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/models', methods=['GET'])
def get_models():
    """Get list of available models."""
    return jsonify({
        "models": get_available_models(),
        "default_model": DEFAULT_MODEL
    })

@app.route('/status', methods=['GET'])
def status():
    # Get model availability
    models = get_available_models()
    available_models = [m for m in models if m["available"]]
    
    return jsonify({
        "documents_count": len(set(chunk["metadata"]["source"] for chunk in document_chunks)),
        "chunks_count": len(document_chunks),
        "ollama_status": "connected" if check_ollama_connection() else "disconnected",
        "available_models": len(available_models),
        "current_model": current_model_id if current_model_id else None,
        "model_loaded": current_model is not None
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