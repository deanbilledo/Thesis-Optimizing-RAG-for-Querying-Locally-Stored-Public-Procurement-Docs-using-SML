import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import gc

MODEL_BASE = "google/gemma-3-1b-it"
ADAPTER_PATH = "./gemma3_1b_procurement_weska_v2"

# Check GPU availability and setup device
if torch.cuda.is_available():
    device = "cuda"
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"💾 Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    # Clear any existing GPU memory
    torch.cuda.empty_cache()
    gc.collect()
else:
    device = "cpu"
    print("⚠️ No GPU detected, using CPU")

# Optimized 4-bit quantization config for better GPU utilization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_storage=torch.uint8,
)

print("📥 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

print("📥 Loading base model with GPU optimization...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="auto",  # Automatically distribute across available GPUs
    low_cpu_mem_usage=True,
    attn_implementation="eager",  # More stable for inference
    trust_remote_code=False
)

if torch.cuda.is_available():
    print(f"💾 GPU memory after model load: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

# Try to load LoRA adapter, fall back to base model if incompatible
try:
    print("📦 Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✅ LoRA adapter loaded successfully!")
except Exception as e:
    print(f"⚠️ Could not load LoRA adapter: {e}")
    print("🔄 Using base model without fine-tuning...")
    model = base_model

# Model is already on GPU via device_map="auto", no need to move again
model.eval()

if torch.cuda.is_available():
    print(f"💾 Final GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"💾 GPU memory available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f}GB")

def chat():
    print("\n🤖 Start chatting with your Philippine procurement expert! (Type 'exit' to quit)")
    print("=" * 70)
    system_prompt = "You are a helpful assistant knowledgeable about Philippine procurement laws."
    history = []
    
    while True:
        try:
            user_input = input("\n🔵 User: ")
            if user_input.strip().lower() == "exit":
                break
                
            messages = [
                {"role": "system", "content": system_prompt},
                *history,
                {"role": "user", "content": user_input}
            ]
            
            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat_prompt, return_tensors="pt")
            
            # Move inputs to the same device as model
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            print("🔄 Generating response...")
            
            with torch.no_grad():
                output = model.generate(
                    **inputs, 
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,  # Enable KV cache for faster generation
                    repetition_penalty=1.1  # Reduce repetition
                )
            
            # Decode only the new tokens (excluding the input)
            input_length = inputs['input_ids'].shape[1]
            new_tokens = output[0][input_length:]
            assistant_reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Clean up any remaining formatting tokens
            if assistant_reply.startswith("<start_of_turn>model"):
                assistant_reply = assistant_reply.replace("<start_of_turn>model", "").strip()
            if assistant_reply.endswith("<end_of_turn>"):
                assistant_reply = assistant_reply.replace("<end_of_turn>", "").strip()
            
            print(f"\n🤖 Assistant: {assistant_reply}")
            print("-" * 70)
            
            # Show GPU memory usage after generation
            if torch.cuda.is_available():
                print(f"💾 GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
            
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": assistant_reply})
            
            # Optional: Clear cache periodically to manage memory
            if torch.cuda.is_available() and len(history) > 10:
                torch.cuda.empty_cache()
                
        except KeyboardInterrupt:
            print("\n👋 Chat interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error during generation: {e}")
            print("🔄 Continuing...")
            continue

if __name__ == "__main__":
    chat()
