#!/usr/bin/env python3
"""
Fine-tune Gemma 3.1-B Instruct on Philippine procurement data
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

# Config
MODEL_NAME = "google/gemma-3-1b-it"
OUTPUT_DIR = "./gemma_1b_procurement_weska_v1"
DATASET_PATH = "train-weska-clean.jsonl"

def setup_environment(use_wandb=False, hf_token=None):
    if hf_token:
        print("🔑 Logging in to Hugging Face...")
        login(token=hf_token)
    if use_wandb:
        wandb.init(project="philippine-procurement-gemma-3b", name="gemma-3-1b-instruct-finetune")

def load_and_prepare_dataset(dataset_path):
    """Load dataset (JSONL) and format as Q → A"""
    formatted_data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)

            # Pure Q&A prompt format
            text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant knowledgeable about Philippine procurement laws.<|eot_id|><|start_header_id|>user<|end_header_id|>

{item['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{item['output']}<|eot_id|>"""

            formatted_data.append({"text": text})

    # Train/val split
    split_idx = int(len(formatted_data) * 0.85)
    train_dataset = Dataset.from_list(formatted_data[:split_idx])
    val_dataset = Dataset.from_list(formatted_data[split_idx:])
    print(f"📊 Dataset loaded: {len(train_dataset)} train / {len(val_dataset)} val")
    return DatasetDict({"train": train_dataset, "validation": val_dataset})

def setup_model_and_tokenizer():
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    print("📥 Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer

def setup_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    setup_environment(args.use_wandb, args.hf_token)

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        return
    dataset = load_and_prepare_dataset(DATASET_PATH)

    model, tokenizer = setup_model_and_tokenizer()
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,                     # ✅ sweet spot (try 5–8 if still weak)
        per_device_train_batch_size=4,          # ✅ small batch = better generalization
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,         # ✅ effective batch = 32 (2 * 16)
        gradient_checkpointing=True,            # ✅ saves VRAM
        warmup_steps=100,                       # ✅ a bit higher since dataset is small
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,                         # ✅ less frequent saves (avoid too many tiny checkpoints)
        eval_strategy="steps",
        eval_steps=100,
        learning_rate=1e-4,                     # ✅ lower than 2e-4 (safer for small data)
        lr_scheduler_type="cosine",             # ✅ smoother decay for small datasets
        fp16=True,
        push_to_hub=False,
        report_to="wandb" if args.use_wandb else "none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        save_total_limit=2,
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
    trainer.train()

    print("\n💾 Saving fine-tuned model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Training complete! Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
