#!/usr/bin/env python3
"""
Fine-tune Microsoft Phi-1.5 on Philippine procurement data
Optimized for 4GB VRAM (RTX 3050)
Dataset format: {"instruction": "...", "output": "..."}
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

# Config
MODEL_NAME = "microsoft/phi-1_5"
OUTPUT_DIR = "./phi_1_5_procurement_weska_v1"
DATASET_PATH = "train-weska-clean.jsonl"

def setup_environment(use_wandb=False, hf_token=None):
    if hf_token:
        print("🔑 Logging in to Hugging Face...")
        login(token=hf_token)
    if use_wandb:
        wandb.init(project="philippine-procurement-phi", name="phi-1.5-finetune")

def load_and_prepare_dataset(dataset_path, max_samples=None):
    """Load dataset (JSONL) and format for Phi-1.5
    
    Phi-1.5 uses simple instruction format without special chat tokens
    Format: Instruct: {instruction}\nOutput: {output}
    """
    formatted_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if max_samples and i >= max_samples:
                break
                
            item = json.loads(line)
            
            # Phi-1.5 instruction format (simple and clean)
            # The model was trained on this format for instruction following
            text = f"""Instruct: {item['instruction']}
Output: {item['output']}"""
            
            formatted_data.append({"text": text})

    # Train/val split
    split_idx = int(len(formatted_data) * 0.85)
    train_dataset = Dataset.from_list(formatted_data[:split_idx])
    val_dataset = Dataset.from_list(formatted_data[split_idx:])
    print(f"📊 Dataset loaded: {len(train_dataset)} train / {len(val_dataset)} val")
    return DatasetDict({"train": train_dataset, "validation": val_dataset})

def setup_model_and_tokenizer():
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    # Phi-1.5 doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print("📥 Loading model in 4-bit (optimized for 4GB VRAM)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_storage=torch.uint8,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,  # Phi models require this
        attn_implementation="eager",  # Use eager attention for stability
    )
    
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def setup_lora_config():
    """LoRA config optimized for Phi-1.5 architecture"""
    return LoraConfig(
        r=8,  # Rank - can go down to 4 if memory is tight
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "dense",  # Phi-1.5 uses "dense" instead of "o_proj"
            "fc1",    # Phi-1.5 MLP layers
            "fc2",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

def tokenize_function(examples, tokenizer, max_length=512):
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
    parser = argparse.ArgumentParser(description="Fine-tune Phi-1.5 on Philippine procurement data")
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size for testing")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    args = parser.parse_args()

    # Clear memory before starting
    clear_memory()
    
    setup_environment(args.use_wandb, args.hf_token)

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return
        
    dataset = load_and_prepare_dataset(DATASET_PATH, args.max_samples)

    model, tokenizer = setup_model_and_tokenizer()
    clear_memory()
    
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        batch_size=100,
        remove_columns=dataset["train"].column_names,
    )
    
    clear_memory()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,          # Effective batch = batch_size * 8
        gradient_checkpointing=True,
        warmup_steps=50,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        fp16=True,
        push_to_hub=False,
        report_to="wandb" if args.use_wandb else "none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=False,
        save_total_limit=2,
        max_grad_norm=1.0,
        optim="adamw_torch",
        group_by_length=False,
        ddp_find_unused_parameters=False,
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
    if torch.cuda.is_available():
        print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"💾 Current GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        print("\n❌ Out of memory! Try:")
        print("  - Reduce --max-length (current: {})".format(args.max_length))
        print("  - Reduce --batch-size (current: {})".format(args.batch_size))
        print("  - Increase gradient_accumulation_steps")
        return
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        return

    print("\n💾 Saving fine-tuned model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Training complete! Model saved to {args.output_dir}")
    
    # Clean up
    clear_memory()

if __name__ == "__main__":
    main()