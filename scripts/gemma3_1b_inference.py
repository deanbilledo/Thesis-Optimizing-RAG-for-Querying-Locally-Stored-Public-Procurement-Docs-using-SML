#!/usr/bin/env python3
"""
Gemma 3 1B Procurement Weska Model Inference Script
==================================================

This script loads and runs inference on the fine-tuned Gemma 3 1B model
with LoRA adapters for procurement-related question answering.

Usage:
    python gemma3_1b_inference.py

Requirements:
    - transformers
    - torch
    - peft
    - accelerate
"""

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

class Gemma3InferenceEngine:
    def __init__(self, model_path="./gemma3_1b_procurement_weska_v2", device=None):
        """
        Initialize the Gemma 3 inference engine.
        
        Args:
            model_path (str): Path to the fine-tuned model directory
            device (str): Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
        print(f"🚀 Initializing Gemma 3 1B Inference Engine...")
        print(f"📁 Model path: {model_path}")
        print(f"💻 Device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the base model, tokenizer, and LoRA adapters."""
        try:
            # Load adapter config to get base model info
            import json
            with open(os.path.join(self.model_path, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config["base_model_name_or_path"]
            print(f"📦 Base model: {base_model_name}")
            
            # Configure quantization for memory efficiency
            if self.device == "cuda":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = None
            
            # Load tokenizer
            print("🔤 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            print("🧠 Loading base model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config if self.device == "cuda" else None,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Load LoRA adapters
            print("🔧 Loading LoRA adapters...")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move to device if not using device_map
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            raise
    
    def format_prompt(self, question, context=None, system_prompt=None):
        """
        Format the input prompt using the chat template.
        
        Args:
            question (str): The question to ask
            context (str, optional): Additional context for the question
            system_prompt (str, optional): System instruction
        
        Returns:
            str: Formatted prompt
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant specialized in procurement and supply chain management. Provide accurate and detailed answers based on the given context."
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        user_content = question
        if context:
            user_content = f"Context: {context}\n\nQuestion: {question}"
        
        messages.append({"role": "user", "content": user_content})
        
        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return formatted_prompt
    
    def generate_response(self, prompt, max_length=512, temperature=0.7, do_sample=True, top_p=0.9):
        """
        Generate a response for the given prompt.
        
        Args:
            prompt (str): The input prompt
            max_length (int): Maximum length of generated text
            temperature (float): Sampling temperature
            do_sample (bool): Whether to use sampling
            top_p (float): Top-p sampling parameter
        
        Returns:
            str: Generated response
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs.input_ids.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            return None
    
    def chat(self, question, context=None, system_prompt=None, **generation_kwargs):
        """
        High-level chat interface.
        
        Args:
            question (str): The question to ask
            context (str, optional): Additional context
            system_prompt (str, optional): System instruction
            **generation_kwargs: Additional generation parameters
        
        Returns:
            str: Model response
        """
        formatted_prompt = self.format_prompt(question, context, system_prompt)
        return self.generate_response(formatted_prompt, **generation_kwargs)

def main():
    """Main function to run interactive inference."""
    print("=" * 60)
    print("🤖 Gemma 3 1B Procurement Weska Inference")
    print("=" * 60)
    
    # Initialize the inference engine
    try:
        engine = Gemma3InferenceEngine()
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return
    
    # Test examples
    test_questions = [
        {
            "question": "What is procurement?",
            "context": None
        },
        {
            "question": "What are the key steps in the procurement process?",
            "context": "We are discussing strategic procurement for large organizations."
        },
        {
            "question": "How do you evaluate suppliers?",
            "context": "We need to select suppliers for IT equipment and services."
        }
    ]
    
    print("\n🧪 Running test questions...")
    print("-" * 40)
    
    for i, test in enumerate(test_questions, 1):
        print(f"\n📋 Test {i}:")
        print(f"Question: {test['question']}")
        if test['context']:
            print(f"Context: {test['context']}")
        
        response = engine.chat(
            question=test['question'],
            context=test['context'],
            max_length=256,
            temperature=0.7
        )
        
        print(f"\n🤖 Response:")
        print(response)
        print("-" * 40)
    
    # Interactive mode
    print("\n🎯 Interactive Mode (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            context = input("📄 Context (optional, press Enter to skip): ").strip()
            context = context if context else None
            
            print("\n🤔 Thinking...")
            response = engine.chat(
                question=question,
                context=context,
                max_length=512,
                temperature=0.7
            )
            
            print(f"\n🤖 Response:")
            print(response)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()