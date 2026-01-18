#!/usr/bin/env python3
"""
Fine-tune Gemma 2-1B-IT on Philippine procurement data
Optimized for 4GB VRAM (RTX 3050)
"""

import json
import torch
import os
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from huggingface_hub import login
import wandb
import argparse
import gc

# Config - Gemma 3-1B-IT
MODEL_NAME = "google/gemma-3-1b-it"
OUTPUT_DIR = "./gemma3_1b_procurement_weska_v3"
DATASET_PATH = "train-weska-clean.jsonl"

def setup_environment(use_wandb=False, hf_token=None):
    if hf_token:
        print("🔑 Logging in to Hugging Face...")
        login(token=hf_token)
    if use_wandb:
        wandb.init(project="philippine-procurement-gemma3", name="gemma-3-1b-instruct-finetune")

def load_and_prepare_dataset(dataset_path, max_samples=None, tokenizer=None, use_system_prompt=True):
    """Load dataset (JSONL) and format using proper Gemma chat template"""
    formatted_data = []
    
    # System prompt for Philippine procurement context
    system_prompt = "You are a helpful assistant knowledgeable about Philippine procurement laws."
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if max_samples and i >= max_samples:
                break
                
            item = json.loads(line)
            
            # Option 1: Use tokenizer's built-in chat template if available
            if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
                try:
                    # Include system prompt if enabled
                    if use_system_prompt:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": item['instruction']},
                            {"role": "assistant", "content": item['output']}
                        ]
                    else:
                        messages = [
                            {"role": "user", "content": item['instruction']},
                            {"role": "assistant", "content": item['output']}
                        ]
                    
                    text = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                except Exception as e:
                    print(f"⚠️  Chat template failed, using manual format: {e}")
                    # Fallback to manual format
                    if use_system_prompt:
                        # System prompt is prepended to first user message in Gemma
                        text = f"""<bos><start_of_turn>user
{system_prompt}
{item['instruction']}<end_of_turn>
<start_of_turn>model
{item['output']}<end_of_turn>
"""
                    else:
                        text = f"""<bos><start_of_turn>user
{item['instruction']}<end_of_turn>
<start_of_turn>model
{item['output']}<end_of_turn>
"""
            else:
                # Manual Gemma 3 chat format following the template
                if use_system_prompt:
                    # System prompt is prepended to first user message
                    text = f"""<bos><start_of_turn>user
{system_prompt}
{item['instruction']}<end_of_turn>
<start_of_turn>model
{item['output']}<end_of_turn>
"""
                else:
                    text = f"""<bos><start_of_turn>user
{item['instruction']}<end_of_turn>
<start_of_turn>model
{item['output']}<end_of_turn>
"""
            
            formatted_data.append({"text": text})

    # Train/val split
    split_idx = int(len(formatted_data) * 0.85)
    train_dataset = Dataset.from_list(formatted_data[:split_idx])
    val_dataset = Dataset.from_list(formatted_data[split_idx:])
    print(f"📊 Dataset loaded: {len(train_dataset)} train / {len(val_dataset)} val")
    if use_system_prompt:
        print(f"💬 System prompt: '{system_prompt}'")
    return DatasetDict({"train": train_dataset, "validation": val_dataset})

def setup_model_and_tokenizer():
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set up padding token properly for Gemma
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print("📥 Loading model in 4-bit (aggressive settings for 4GB VRAM)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Use fp16 instead of bfloat16
        bnb_4bit_quant_storage=torch.uint8,   # More aggressive quantization
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,  # Changed from bfloat16
        low_cpu_mem_usage=True,
        trust_remote_code=False,  # Gemma 3 doesn't need trust_remote_code
        attn_implementation="eager",  # Use eager attention for better memory efficiency
    )
    
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def setup_lora_config():
    """More aggressive LoRA for memory efficiency"""
    return LoraConfig(
        r=4,  # Reduced from 8 to save memory
        lora_alpha=8,  # Reduced proportionally
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,  # Reduced dropout
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=None,  # Don't save additional modules
    )

def tokenize_function(examples, tokenizer, max_length=256):  # Reduced max length
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

def clear_memory():
    """Aggressive memory clearing"""
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--epochs", type=int, default=8)  # Reduced default
    parser.add_argument("--batch-size", type=int, default=4)  # Very small batch
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=256)  # Much shorter sequences
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for testing")
    parser.add_argument("--no-system-prompt", action="store_true", help="Disable system prompt")
    args = parser.parse_args()

    # Clear memory before starting
    clear_memory()
    
    setup_environment(args.use_wandb, args.hf_token)

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return
    
    # Load tokenizer first to use chat template
    print("📥 Loading tokenizer for dataset preparation...")
    temp_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_and_prepare_dataset(
        DATASET_PATH, 
        args.max_samples, 
        temp_tokenizer,
        use_system_prompt=not args.no_system_prompt
    )

    model, tokenizer = setup_model_and_tokenizer()
    clear_memory()
    
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        batch_size=100,  # Process in small batches
        remove_columns=dataset["train"].column_names,
    )
    
    clear_memory()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,          # Minimum batch size
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,          # Higher accumulation to simulate larger batches
        gradient_checkpointing=True,
        warmup_steps=50,
        logging_steps=25,
        save_strategy="epoch",                  # Save less frequently
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        fp16=True,                              # Use fp16 for memory efficiency
        push_to_hub=False,
        report_to="wandb" if args.use_wandb else "none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,               # No parallel data loading
        metric_for_best_model="eval_loss",
        load_best_model_at_end=False,           # Skip to save memory
        save_total_limit=1,                     # Keep only 1 checkpoint
        max_grad_norm=1.0,                      # Gradient clipping
        optim="adamw_torch",                    # Use PyTorch optimizer
        group_by_length=False,                  # Disable to save memory
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\n🚀 Starting training...")
    print(f"💾 Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"💾 Current GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("❌ Out of memory! Try reducing batch size or sequence length further.")
        return

    print("\n💾 Saving fine-tuned model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Training complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()