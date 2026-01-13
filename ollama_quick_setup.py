#!/usr/bin/env python3
"""
Simple Ollama model creator using existing merged Gemma model
"""

import os
import subprocess
from pathlib import Path

# Configuration
MERGED_MODEL_PATH = "./gemma3_v2_merged_model"  # Your existing merged model
OLLAMA_MODEL_NAME = "gemma3-procurement-weska"

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

def check_ollama():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Ollama version: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Ollama not found!")
        return False

def install_ollama_instructions():
    """Print installation instructions"""
    print("""
📋 To install Ollama:

Windows (Recommended):
1. Download from: https://ollama.ai/download/windows
2. Run the installer
3. Restart your terminal

Alternative (winget):
winget install Ollama.Ollama

After installation, run this script again.
""")

def create_ollama_model():
    """Create the Ollama model"""
    if not os.path.exists(MERGED_MODEL_PATH):
        print(f"❌ Merged model not found at: {MERGED_MODEL_PATH}")
        return False
    
    print(f"📁 Using merged model from: {MERGED_MODEL_PATH}")
    
    # Create Modelfile
    modelfile_path = create_modelfile()
    
    if not check_ollama():
        install_ollama_instructions()
        print(f"\n📁 Files ready for manual Ollama setup:")
        print(f"   - Model: {MERGED_MODEL_PATH}")
        print(f"   - Modelfile: ./Modelfile")
        print(f"\nOnce Ollama is installed, run:")
        print(f"   ollama create {OLLAMA_MODEL_NAME} -f Modelfile")
        return False
    
    # Create the model in Ollama
    print(f"🔄 Creating Ollama model '{OLLAMA_MODEL_NAME}'...")
    
    cmd = ["ollama", "create", OLLAMA_MODEL_NAME, "-f", modelfile_path]
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Successfully created Ollama model: {OLLAMA_MODEL_NAME}")
        print(f"🎉 You can now use it with: ollama run {OLLAMA_MODEL_NAME}")
        
        # List models to confirm
        print(f"\n📋 Available Ollama models:")
        subprocess.run(["ollama", "list"])
        
        # Quick test
        print(f"\n🧪 Testing model (quick response)...")
        test_cmd = ["ollama", "run", OLLAMA_MODEL_NAME, "--", "Hello, what do you do?"]
        test_result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=15)
        
        if test_result.returncode == 0:
            print("✅ Model test successful!")
            print("Response:", test_result.stdout.strip()[:150] + "...")
        else:
            print("⚠️  Model created but test failed (this is normal)")
            
        return True
    else:
        print(f"❌ Failed to create Ollama model:")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False

def main():
    print("🤖 Gemma 3 1B Procurement → Ollama (Quick Setup)")
    print("=" * 50)
    
    success = create_ollama_model()
    
    if success:
        print(f"\n🎉 SUCCESS! Your model is ready!")
        print(f"\n📝 Usage examples:")
        print(f"   ollama run {OLLAMA_MODEL_NAME}")
        print(f"   ollama run {OLLAMA_MODEL_NAME} 'What are the procurement requirements for government contracts?'")
        
        print(f"\n💡 Integration examples:")
        print(f"   # Python with ollama package")
        print(f"   import ollama")
        print(f"   response = ollama.chat(model='{OLLAMA_MODEL_NAME}', messages=[{{'role': 'user', 'content': 'Your question'}}])")
        
        print(f"\n   # REST API (default port 11434)")
        print(f"   curl http://localhost:11434/api/generate -d '{{\"model\": \"{OLLAMA_MODEL_NAME}\", \"prompt\": \"Your question\"}}'")
    else:
        print(f"\n❌ Setup incomplete. Follow the instructions above to finish manually.")

if __name__ == "__main__":
    main()