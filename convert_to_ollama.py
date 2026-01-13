#!/usr/bin/env python3
"""
Convert Gemma 3 1B fine-tuned model to Ollama format
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
BASE_MODEL = "google/gemma-3-1b-it"  # Correct base model from adapter config
FINETUNED_MODEL_PATH = "./gemma3_1b_procurement_weska_final"
MERGED_MODEL_PATH = "./gemma3_merged_for_ollama"
OLLAMA_MODEL_NAME = "gemma3-procurement-weska"

def merge_and_save_model():
    """Merge PEFT adapter with base model and save"""
    print("🔄 Loading base model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="auto"
    )
    
    print("🔄 Loading fine-tuned adapter...")
    
    # Load PEFT model
    model = PeftModel.from_pretrained(
        base_model,
        FINETUNED_MODEL_PATH,
        torch_dtype="auto"
    )
    
    print("🔄 Merging adapter with base model...")
    
    # Merge and unload adapter
    merged_model = model.merge_and_unload()
    
    print(f"💾 Saving merged model to {MERGED_MODEL_PATH}...")
    
    # Save merged model
    merged_model.save_pretrained(
        MERGED_MODEL_PATH,
        safe_serialization=True
    )
    tokenizer.save_pretrained(MERGED_MODEL_PATH)
    
    print("✅ Model merged and saved successfully!")
    return MERGED_MODEL_PATH

def create_modelfile():
    """Create Ollama Modelfile"""
    modelfile_content = f"""FROM {MERGED_MODEL_PATH}

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

TEMPLATE \"\"\"<start_of_turn>user
{{{{ prompt }}}}<end_of_turn>
<start_of_turn>model
\"\"\"

SYSTEM \"\"\"You are a helpful assistant specialized in procurement and government contracting. You provide accurate, detailed answers about procurement processes, regulations, and best practices based on your training data.\"\"\"
"""
    
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)
    
    print("📝 Modelfile created successfully!")
    return "Modelfile"

def convert_to_ollama():
    """Convert model to Ollama format"""
    print("🚀 Starting Ollama conversion process...")
    
    try:
        # Step 1: Merge PEFT model
        merged_path = merge_and_save_model()
        
        # Step 2: Create Modelfile
        modelfile_path = create_modelfile()
        
        # Step 3: Create Ollama model
        print(f"🔄 Creating Ollama model '{OLLAMA_MODEL_NAME}'...")
        
        # Check if ollama is available
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Ollama version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Ollama not found! Please install Ollama first:")
            print("   Download from: https://ollama.ai/download")
            return False
        
        # Create the model in Ollama
        cmd = ["ollama", "create", OLLAMA_MODEL_NAME, "-f", modelfile_path]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully created Ollama model: {OLLAMA_MODEL_NAME}")
            print(f"🎉 You can now use it with: ollama run {OLLAMA_MODEL_NAME}")
            
            # Test the model
            print("\n🧪 Testing the model...")
            test_cmd = ["ollama", "run", OLLAMA_MODEL_NAME, 
                       "What is the procurement process for government contracts?"]
            
            test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
            if test_result.returncode == 0:
                print("✅ Model test successful!")
                print("Response preview:", test_result.stdout[:200] + "...")
            else:
                print("⚠️  Model created but test failed:", test_result.stderr)
                
        else:
            print(f"❌ Failed to create Ollama model:")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        return False
    
    return True

def install_ollama_instructions():
    """Print instructions for installing Ollama"""
    print("""
📋 To install Ollama:

Windows:
1. Download from: https://ollama.ai/download/windows
2. Run the installer
3. Restart your terminal
4. Verify with: ollama --version

Alternative (with winget):
winget install Ollama.Ollama

Linux/macOS:
curl -fsSL https://ollama.ai/install.sh | sh

After installation, run this script again.
""")

def main():
    print("🤖 Gemma 3 1B Procurement Model → Ollama Converter")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists(FINETUNED_MODEL_PATH):
        print(f"❌ Fine-tuned model not found at: {FINETUNED_MODEL_PATH}")
        return
    
    # Check if ollama is available
    try:
        subprocess.run(["ollama", "--version"], 
                      capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ollama not found!")
        install_ollama_instructions()
        
        # Still create merged model for manual use
        print("\n🔄 Creating merged model anyway (for manual Ollama setup)...")
        merge_and_save_model()
        create_modelfile()
        
        print(f"\n📁 Files created:")
        print(f"   - Merged model: {MERGED_MODEL_PATH}")
        print(f"   - Modelfile: ./Modelfile")
        print(f"\nOnce Ollama is installed, run:")
        print(f"   ollama create {OLLAMA_MODEL_NAME} -f Modelfile")
        return
    
    # Convert to Ollama
    success = convert_to_ollama()
    
    if success:
        print(f"\n🎉 Success! Your model is now available as: {OLLAMA_MODEL_NAME}")
        print(f"\nUsage examples:")
        print(f"   ollama run {OLLAMA_MODEL_NAME}")
        print(f"   ollama run {OLLAMA_MODEL_NAME} 'What are the procurement requirements?'")
        
        # List available models
        print(f"\n📋 Available Ollama models:")
        subprocess.run(["ollama", "list"])
        
    else:
        print(f"\n❌ Conversion failed. Check the errors above.")

if __name__ == "__main__":
    main()