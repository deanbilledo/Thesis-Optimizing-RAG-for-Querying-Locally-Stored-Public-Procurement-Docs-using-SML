#!/usr/bin/env python3
"""
Fine-tune Microsoft Phi-1.5 on Philippine procurement data - V3 OPTIMIZED
Optimized for 4GB VRAM with improved training strategies
Dataset format: {"instruction": "...", "output": "..."}

Key improvements in V3:
- Better LoRA configuration with higher rank
- Optimized learning rate schedule with warmup ratio
- Improved tokenization with dynamic padding
- Better gradient accumulation strategy
- Enhanced memory management
- QLoRA optimizations for better convergence
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
    BitsAndBytesConfig,
    EarlyStoppingCallback
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
OUTPUT_DIR = "./phi_1_5_procurement_weska_v3"
DATASET_PATH = "train-weska-clean.jsonl"

def setup_environment(use_wandb=False, hf_token=None):
    if hf_token:
        print("🔑 Logging in to Hugging Face...")
        login(token=hf_token)
    if use_wandb:
        wandb.init(
            project="philippine-procurement-phi",
            name="phi-1.5-finetune-v3",
            config={
                "model": MODEL_NAME,
                "version": "v3-optimized"
            }
        )

def load_and_prepare_dataset(dataset_path, max_samples=None):
    """Load dataset with improved formatting for instruction following
    
    V3 improvements:
    - Clearer instruction/output separation
    - Better prompt structure for Phi-1.5
    """
    formatted_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if max_samples and i >= max_samples:
                break
                
            item = json.loads(line)
            
            # Improved format with explicit markers
            text = f"""Instruct: {item['instruction'].strip()}
Output: {item['output'].strip()}"""
            
            formatted_data.append({"text": text})

    # Better train/val split (90/10 for larger datasets)
    split_idx = int(len(formatted_data) * 0.9)
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
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print("📥 Loading model in 4-bit with optimized quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,  # Nested quantization for better memory
        bnb_4bit_quant_type="nf4",  # NormalFloat4 - best for fine-tuning
        bnb_4bit_compute_dtype=torch.bfloat16,  # Changed to bfloat16 for better stability
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Match compute dtype
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    # Prepare for k-bit training with gradient checkpointing
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True
    )
    
    # Enable input gradients for embeddings
    model.config.use_cache = False
    
    return model, tokenizer

def setup_lora_config():
    """V3: Improved LoRA config for better parameter efficiency
    
    Changes:
    - Increased rank from 8 to 16 for better capacity
    - Adjusted alpha for optimal scaling
    - Added more target modules for comprehensive adaptation
    """
    return LoraConfig(
        r=16,  # Increased rank for better expressiveness
        lora_alpha=32,  # 2x rank for optimal scaling
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "dense",  # Output projection
            "fc1",    # MLP up projection
            "fc2",    # MLP down projection
        ],
        lora_dropout=0.1,  # Increased dropout for regularization
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )

def tokenize_function(examples, tokenizer, max_length=512):
    """V3: Improved tokenization with better handling"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,  # Dynamic padding in collator
        return_attention_mask=True,
    )
    
    # Add labels for causal LM
    result["labels"] = result["input_ids"].copy()
    return result

def clear_memory():
    """Aggressive memory clearing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def compute_metrics(eval_pred):
    """Simple perplexity computation for monitoring"""
    predictions, labels = eval_pred
    # This is just for logging purposes during eval
    return {}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-1.5 V3 - Optimized")
    parser.add_argument("--use-wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    args = parser.parse_args()

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
    
    print("\n📊 Trainable parameters:")
    model.print_trainable_parameters()

    print("\n🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        batch_size=100,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    
    clear_memory()

    # V3: Optimized training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,  # Effective batch = batch_size * grad_accum
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # V3: Improved learning rate schedule
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,  # Warmup for 10% of training
        lr_scheduler_type="cosine",  # Smooth decay
        weight_decay=0.01,  # L2 regularization
        max_grad_norm=1.0,  # Gradient clipping
        
        # Precision and optimization
        bf16=True,  # BFloat16 for better stability than FP16
        optim="paged_adamw_8bit",  # 8-bit optimizer for memory efficiency
        
        # Logging and evaluation
        logging_steps=5,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Memory optimizations
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        ddp_find_unused_parameters=False,
        
        # Reporting
        report_to="wandb" if args.use_wandb else "none",
        run_name="phi-1.5-v3",
        push_to_hub=False,
    )

    # V3: Use dynamic padding for efficiency
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # Optimize for tensor cores
    )

    # V3: Add early stopping callback
    callbacks = []
    if len(tokenized_dataset["validation"]) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print("\n🚀 Starting V3 optimized training...")
    if torch.cuda.is_available():
        print(f"💾 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"💾 Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        print(f"💾 Reserved: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    
    print(f"\n📈 Training config:")
    print(f"   Effective batch size: {args.batch_size * args.grad_accum}")
    print(f"   Total steps: ~{len(tokenized_dataset['train']) // (args.batch_size * args.grad_accum) * args.epochs}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   LoRA rank: 16")
    
    try:
        trainer.train()
        print("\n✅ Training completed successfully!")
    except torch.cuda.OutOfMemoryError:
        print("\n❌ Out of memory! Try:")
        print(f"  - Reduce --max-length (current: {args.max_length})")
        print(f"  - Reduce --batch-size (current: {args.batch_size})")
        print(f"  - Increase --grad-accum (current: {args.grad_accum})")
        print(f"  - Reduce LoRA rank in setup_lora_config()")
        return
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n💾 Saving fine-tuned model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metrics
    if trainer.state.log_history:
        with open(os.path.join(args.output_dir, "training_log.json"), "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
    
    print(f"\n✅ Model saved to {args.output_dir}")
    print("\n📊 Final metrics:")
    if trainer.state.best_metric is not None:
        print(f"   Best eval loss: {trainer.state.best_metric:.4f}")
    
    clear_memory()
    
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()