"""
Procurement Document Fine-tuning with LoRA
Optimized for RTX 3050 Laptop GPU (4GB VRAM)
"""

import json
import torch
import gc
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import warnings
warnings.filterwarnings("ignore")

def setup_quantization_config():
    """Setup 4-bit quantization for memory efficiency"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

def load_model_and_tokenizer(model_name):
    """Load model with quantization and tokenizer"""
    print(f"Loading model: {model_name}")
    
    # Setup quantization
    bnb_config = setup_quantization_config()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def setup_lora_config():
    """Setup LoRA configuration"""
    return LoraConfig(
        r=16,                    # Rank
        lora_alpha=32,           # Alpha scaling
        target_modules=[         # Target modules for Phi-3
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

def load_jsonl_dataset(file_path):
    """Load JSONL dataset"""
    print(f"Loading dataset from: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} examples")
    return data

def format_training_examples(examples, tokenizer, max_length=512):
    """Format examples for instruction following"""
    formatted_examples = []
    
    for example in examples:
        # Create instruction format
        instruction_text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        
        # Tokenize and check length
        tokens = tokenizer.encode(instruction_text, add_special_tokens=True)
        
        if len(tokens) <= max_length:
            formatted_examples.append(instruction_text)
        else:
            # Truncate if too long
            truncated_tokens = tokens[:max_length-1] + [tokenizer.eos_token_id]
            truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)
            formatted_examples.append(truncated_text)
    
    return formatted_examples

def tokenize_dataset(examples, tokenizer, max_length=512):
    """Tokenize the dataset"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

def main():
    print("=" * 50)
    print("PROCUREMENT DOCUMENT FINE-TUNING WITH LORA")
    print("=" * 50)
    
    # Configuration
    MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
    DATASET_FILE = "key-responses.jsonl"
    OUTPUT_DIR = "./procurement-lora-checkpoint"
    MERGED_OUTPUT_DIR = "./procurement-merged"
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    
    # Step 1: Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    # Step 2: Setup LoRA
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)
    
    print(f"LoRA trainable parameters: {model.num_parameters()}")
    model.print_trainable_parameters()
    
    # Step 3: Load and prepare dataset
    raw_data = load_jsonl_dataset(DATASET_FILE)
    formatted_texts = format_training_examples(raw_data, tokenizer)
    
    # Create dataset
    dataset = Dataset.from_dict({"text": formatted_texts})
    tokenized_dataset = dataset.map(
        lambda x: tokenize_dataset(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Step 4: Setup training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,        # Small batch for 4GB VRAM
        gradient_accumulation_steps=8,         # Effective batch size = 8
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,                            # Mixed precision
        logging_steps=10,
        save_steps=50,
        save_strategy="steps",
        evaluation_strategy="no",
        remove_unused_columns=False,
        dataloader_pin_memory=False,          # Reduce memory usage
        gradient_checkpointing=True,
        dataloader_num_workers=0,             # Reduce memory usage
        group_by_length=False,                # Reduce memory usage
        report_to=None,                       # Disable wandb/tensorboard
        run_name="procurement-lora-training"
    )
    
    # Step 5: Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
        pad_to_multiple_of=8
    )
    
    # Step 6: Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Step 7: Start training
    print("\n" + "=" * 30)
    print("STARTING TRAINING")
    print("=" * 30)
    
    try:
        trainer.train()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
        return
    
    # Step 8: Save LoRA checkpoint
    print("Saving LoRA checkpoint...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Step 9: Merge LoRA weights with base model
    print("\n" + "=" * 30)
    print("MERGING LORA WEIGHTS")
    print("=" * 30)
    
    # Clear memory
    del trainer, model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load base model again (without quantization for merging)
    print("Loading base model for merging...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA model
    print("Loading LoRA weights...")
    merged_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    
    # Merge and unload
    print("Merging weights...")
    merged_model = merged_model.merge_and_unload()
    
    # Save merged model
    print(f"Saving merged model to: {MERGED_OUTPUT_DIR}")
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    tokenizer.save_pretrained(MERGED_OUTPUT_DIR)
    
    print("Model merging completed!")
    
    # Step 10: Test inference
    print("\n" + "=" * 30)
    print("TESTING INFERENCE")
    print("=" * 30)
    
    # Test the merged model
    test_instruction = "Who is the winning bidder?"
    test_input = "Notice of Award: XYZ Construction Corp."
    test_prompt = f"### Instruction:\n{test_instruction}\n\n### Input:\n{test_input}\n\n### Response:\n"
    
    print(f"Test prompt: {test_prompt}")
    
    # Tokenize and generate
    inputs = tokenizer.encode(test_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = merged_model.generate(
            inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the response part
    response_start = response.find("### Response:\n") + len("### Response:\n")
    generated_response = response[response_start:].strip()
    
    print(f"Generated response: {generated_response}")
    
    print("\n" + "=" * 50)
    print("FINE-TUNING COMPLETE!")
    print(f"LoRA checkpoint saved to: {OUTPUT_DIR}")
    print(f"Merged model saved to: {MERGED_OUTPUT_DIR}")
    print("=" * 50)
    
    # Clean up
    del merged_model, base_model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
