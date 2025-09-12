import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------------
# Config
# ------------------
model_id = "meta-llama/Llama-3.2-3B-Instruct"
data_file = "key-responses.jsonl"
output_dir = "./lora-llama3-procurement"

# ------------------
# Load model + tokenizer
# ------------------
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Add pad token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cpu"
)
# Skip quantization for CPU training
# model = prepare_model_for_kbit_training(model)

# ------------------
# Apply LoRA
# ------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# ------------------
# Load dataset
# ------------------
dataset = load_dataset("json", data_files={"train": data_file})

def format_sample(example):
    # Create training text with instruction, input, and expected output
    text = f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse: {example['output']}"
    return {"text": text}

dataset = dataset["train"].map(format_sample)

def tokenize(batch):
    # Tokenize the formatted text
    encoded = tokenizer(
        batch["text"],
        truncation=True,
        padding=False,  # DataCollator will handle padding
        max_length=512
    )
    
    # For causal LM, labels are the same as input_ids
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# ------------------
# Training setup
# ------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal LM, not masked LM
)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,       # Small batch for CPU training
    gradient_accumulation_steps=8,
    num_train_epochs=1,                  # Reduced for CPU training
    learning_rate=2e-4,
    fp16=False,                          # Disable fp16 for CPU
    save_strategy="epoch",
    logging_steps=20,
    no_cuda=True                         # Force CPU training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

# ------------------
# Train
# ------------------
trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
