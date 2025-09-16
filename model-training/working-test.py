from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import torch

def test_with_and_without_lora():
    base_model_path = r"C:\thesis-model-phi\Llama-3.2-3B"
    lora_adapter_path = r"./lora-procurement-new"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("📥 Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="cpu",
        dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # Test prompts
    general_prompt = "What is artificial intelligence?"
    domain_prompt = "What is the short title of Republic Act No. 9184?"
    
    print("\n🧪 Testing BASE MODEL ONLY:")
    base_generator = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        do_sample=True
    )
    
    print("\n--- General Knowledge Test ---")
    output = base_generator(general_prompt, max_new_tokens=100)
    print(f"Base model: {output[0]['generated_text'][len(general_prompt):].strip()}")
    
    print("\n--- Domain Knowledge Test ---")
    output = base_generator(domain_prompt, max_new_tokens=100)
    print(f"Base model: {output[0]['generated_text'][len(domain_prompt):].strip()}")
    
    # Now add LoRA adapter
    print("\n📥 Loading LoRA adapter...")
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    print("\n🧪 Testing BASE + LORA MODEL:")
    lora_generator = pipeline(
        "text-generation",
        model=lora_model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        do_sample=True
    )
    
    print("\n--- General Knowledge Test ---")
    output = lora_generator(general_prompt, max_new_tokens=100)
    print(f"LoRA model: {output[0]['generated_text'][len(general_prompt):].strip()}")
    
    print("\n--- Domain Knowledge Test ---")
    output = lora_generator(domain_prompt, max_new_tokens=100)
    print(f"LoRA model: {output[0]['generated_text'][len(domain_prompt):].strip()}")

if __name__ == "__main__":
    test_with_and_without_lora()