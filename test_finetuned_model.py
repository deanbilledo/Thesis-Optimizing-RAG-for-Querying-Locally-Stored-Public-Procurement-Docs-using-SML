#!/usr/bin/env python3
"""
Test script to verify the fine-tuned TinyLlama model works correctly.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# Configuration
FINETUNED_MODEL_PATH = "model-training/tinyllama_procurement_20250830_104738"
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

def test_finetuned_model():
    """Test the fine-tuned model with procurement-specific queries."""
    
    print("🔄 Loading fine-tuned TinyLlama model...")
    
    try:
        # Check if model exists
        if not os.path.exists(FINETUNED_MODEL_PATH):
            print(f"❌ Model not found at: {FINETUNED_MODEL_PATH}")
            return False
        
        # Check GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {device}")
        
        # Load tokenizer
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load base model
        print("🧠 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        
        if device == "cpu":
            base_model = base_model.to(device)
        
        # Load LoRA weights
        print("🎯 Loading LoRA weights...")
        model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
        model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Test queries
        test_queries = [
            "What is Republic Act 9184?",
            "Define procurement in government context.",
            "What are the requirements for public bidding?",
            "Explain the role of a Bids and Awards Committee.",
            "What is emergency procurement?"
        ]
        
        print("\n" + "="*60)
        print("🧪 TESTING FINE-TUNED MODEL")
        print("="*60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📋 Test {i}/5: {query}")
            print("-" * 50)
            
            # Create prompt
            prompt = f"Question: {query}\nAnswer: "
            
            # Tokenize and move to correct device
            inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = inputs.to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    attention_mask=torch.ones_like(inputs)  # Add attention mask
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            print(f"🤖 Response: {generated_text if generated_text else 'No response generated'}")
        
        print("\n" + "="*60)
        print("✅ Testing completed successfully!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_base_model():
    """Compare fine-tuned vs base model on a simple query."""
    
    print("\n🔍 COMPARISON: Fine-tuned vs Base Model")
    print("="*60)
    
    query = "What is Republic Act 9184?"
    prompt = f"Question: {query}\nAnswer: "
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test base model
        print("\n🏗️  Base Model Response:")
        print("-" * 30)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = base_model.generate(
                inputs,
                max_length=inputs.shape[1] + 80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        base_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        base_generated = base_response[len(prompt):].strip()
        print(f"📝 {base_generated}")
        
        # Test fine-tuned model
        print("\n🎯 Fine-tuned Model Response:")
        print("-" * 30)
        
        finetuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
        finetuned_model.eval()
        
        with torch.no_grad():
            outputs = finetuned_model.generate(
                inputs,
                max_length=inputs.shape[1] + 80,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        ft_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ft_generated = ft_response[len(prompt):].strip()
        print(f"📝 {ft_generated}")
        
        print("\n✅ Comparison completed!")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")

if __name__ == "__main__":
    print("🚀 Fine-tuned TinyLlama Model Tester")
    print("="*60)
    
    # Test the fine-tuned model
    success = test_finetuned_model()
    
    if success:
        # Compare with base model
        compare_with_base_model()
    
    print("\n🎉 Testing completed!")
