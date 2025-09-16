from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import torch
import json
import wandb

# --------------------------
# 1. Clean and load dataset
# --------------------------
def clean_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        good, bad = 0, 0
        for i, line in enumerate(infile, 1):
            try:
                json.loads(line)
                outfile.write(line)
                good += 1
            except json.JSONDecodeError as e:
                print(f"❌  JSON at line {i}: {e}")
                bad += 1
        print(f"✅ Cleaned {good} lines, skipped {bad} bad lines from {input_path}")

# Clean train.jsonl
clean_jsonl("train.jsonl", "train_clean.jsonl")

# Load dataset
dataset = load_dataset("json", data_files="train_clean.jsonl")["train"]

def format_example(example):
    if example.get("input"):
        prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nAnswer:"
    else:
        prompt = f"Instruction: {example['instruction']}\nAnswer:"
    return {"text": prompt + " " + example["output"]}

dataset = dataset.map(format_example)

# Split 90/10
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_data = dataset["train"]
eval_data = dataset["test"]

# --------------------------
# 2. Load base model w/ quantization
# --------------------------
model_name = r"C:\thesis-model-phi\Llama-3.2-3B"  # local path

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",               # automatically split layers GPU/CPU
    quantization_config=bnb_config,
    low_cpu_mem_usage=True
)

# --------------------------
# 3. Apply LoRA
# --------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# --------------------------
# 4. Tokenization
# --------------------------
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    tokens["labels"] = tokens["input_ids"].copy()   # 👈 important for loss computation
    return tokens


train_data = train_data.map(tokenize_function, batched=True, remove_columns=train_data.column_names)
eval_data = eval_data.map(tokenize_function, batched=True, remove_columns=eval_data.column_names)

# --------------------------
# 5. Training config
# --------------------------
training_args = TrainingArguments(
    output_dir="./llama3-procurement",
    per_device_train_batch_size=2,   # keep small for 4GB VRAM
    gradient_accumulation_steps=8,  # simulate larger batch
    learning_rate=2e-4,
    max_steps=300,  
    logging_steps=25,
    save_strategy="steps",
    eval_strategy="steps",           # <-- works in your version    # <-- fixed key
    eval_steps=100,                   # <-- works in your version    # <-- fixed key
    fp16=torch.cuda.is_available(),
    save_total_limit=2,
    report_to=["wandb"],             # enable W&B logging
    run_name="llama3-procurement"
)

# --------------------------
# 6. Trainer
# --------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=tokenizer   # replaces deprecated `tokenizer`
)

# --------------------------
# 7. Train
# --------------------------
trainer.train()

# --------------------------
# 8. Save adapter
# --------------------------
model.save_pretrained("./lora-procurement-new")
