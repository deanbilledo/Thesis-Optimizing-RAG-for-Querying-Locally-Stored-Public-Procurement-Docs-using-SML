import json
import torch
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Check if required packages are installed
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        TrainingArguments, 
        Trainer,
        DataCollatorForLanguageModeling,
        TrainerCallback
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
    from datasets import Dataset
    from trl import SFTTrainer
    import sys
except ImportError as e:
    print(f"Missing required packages. Install with:")
    print("pip install transformers datasets peft accelerate bitsandbytes trl")
    print(f"Error: {e}")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

class DetailedProgressCallback(TrainerCallback):
    """Custom callback to show detailed training progress."""
    
    def __init__(self):
        self.start_time = None
        self.step_times = []
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        
        if self.start_time is not None:
            # Calculate timing statistics
            elapsed_time = current_time - self.start_time
            current_step = state.global_step
            total_steps = state.max_steps
            
            # Calculate steps per second
            if elapsed_time > 0:
                steps_per_sec = current_step / elapsed_time
                it_per_s = steps_per_sec
            else:
                it_per_s = 0.0
            
            # Calculate ETA
            if it_per_s > 0:
                remaining_steps = total_steps - current_step
                eta_seconds = remaining_steps / it_per_s
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                eta_secs = int(eta_seconds % 60)
                eta_str = f"{eta_hours}:{eta_minutes:02d}:{eta_secs:02d}"
            else:
                eta_str = "??:??:??"
            
            # Calculate elapsed time
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_minutes = int((elapsed_time % 3600) // 60)
            elapsed_secs = int(elapsed_time % 60)
            elapsed_str = f"{elapsed_hours}:{elapsed_minutes:02d}:{elapsed_secs:02d}"
            
            # Calculate current epoch
            if hasattr(state, 'epoch'):
                current_epoch = state.epoch
            else:
                # Estimate epoch based on steps
                steps_per_epoch = total_steps / args.num_train_epochs
                current_epoch = current_step / steps_per_epoch
            
            # Create progress bar
            progress_bar_width = 50
            progress = current_step / total_steps
            filled_length = int(progress_bar_width * progress)
            bar = '█' * filled_length + '▒' * (progress_bar_width - filled_length)
            
            # Format the status line exactly like the example
            status_line = f"{current_step}/{total_steps} {elapsed_str} < {eta_str}, {it_per_s:.2f} it/s, Epoch {current_epoch:.2f}/{args.num_train_epochs}"
            
            # Print progress bar with status
            print(f"\r[{bar}] {status_line}", end="", flush=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Handle logging without interfering with progress bar - COMPLETELY SAFE VERSION."""
        if logs and state.global_step > 0:
            # Print loss information on a new line occasionally
            if state.global_step % (args.logging_steps * 5) == 0:  # Every 5 logging steps
                loss = logs.get('train_loss', 'N/A')
                lr = logs.get('learning_rate', 'N/A')
                
                # SAFE formatting - no f-string formatting on potentially string values
                loss_str = "N/A"
                lr_str = "N/A"
                
                # Handle loss safely
                if loss != 'N/A':
                    try:
                        loss_float = float(loss)
                        loss_str = f"{loss_float:.4f}"
                    except (ValueError, TypeError):
                        loss_str = str(loss)
                
                # Handle learning rate safely  
                if lr != 'N/A':
                    try:
                        lr_float = float(lr)
                        lr_str = f"{lr_float:.2e}"
                    except (ValueError, TypeError):
                        lr_str = str(lr)
                
                # Use simple string concatenation to avoid any f-string issues
                log_msg = f"\nStep {state.global_step}: Loss={loss_str}, LR={lr_str}"
                print(log_msg)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Training completion message."""
        if self.start_time is not None:
            total_time = time.time() - self.start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            print(f"\n\n" + "="*80)
            print("TRAINING COMPLETED!")
            print(f"Total training time: {hours}:{minutes:02d}:{seconds:02d}")
            print(f"Total steps: {state.global_step}")
            print(f"Final epoch: {state.epoch:.2f}")
            print("="*80)

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning parameters."""
    
    # Model settings - Using TinyLlama for your setup
    model_name: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    jsonl_file: str = "pdfs_dataset.jsonl"
    output_dir: str = "tinyllama_procurement_finetuned"
    
    # LoRA settings for efficient training
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training hyperparameters optimized for RTX 3050 (4GB)
    num_epochs: int = 3
    batch_size: int = 1  # Small batch size for 4GB GPU
    gradient_accumulation_steps: int = 8  # Compensate with more accumulation
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    max_seq_length: int = 1024  # Reduced for memory efficiency
    
    # System settings
    use_fp16: bool = True
    use_gradient_checkpointing: bool = True
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10

class TinyLlamaFineTuner:
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.dataset = None
        
    def load_jsonl_data(self) -> List[Dict[str, Any]]:
        """Load and parse JSONL file."""
        logger.info(f"Loading data from {self.config.jsonl_file}")
        
        if not Path(self.config.jsonl_file).exists():
            raise FileNotFoundError(f"JSONL file not found: {self.config.jsonl_file}")
        
        data = []
        with open(self.config.jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    data.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(data)} PDF documents")
        return data
    
    def prepare_training_data(self, pdf_data: List[Dict[str, Any]]) -> List[str]:
        """Convert PDF data into training format for TinyLlama."""
        training_texts = []
        
        for record in pdf_data:
            filename = record.get('filename', 'document')
            text = record.get('text', '').strip()
            
            if not text or len(text) < 50:
                continue
            
            # Create instruction-following format for TinyLlama
            # Using a simple format that works well with smaller models
            prompt = f"Document: {filename}\nContent: {text}\n"
            
            # Split long texts into chunks for better training
            if len(prompt) > self.config.max_seq_length * 3:  # Rough token estimate
                chunks = self._split_text_into_chunks(text, filename)
                training_texts.extend(chunks)
            else:
                training_texts.append(prompt)
        
        logger.info(f"Created {len(training_texts)} training examples")
        return training_texts
    
    def _split_text_into_chunks(self, text: str, filename: str, chunk_size: int = 800) -> List[str]:
        """Split long text into smaller chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            prompt = f"Document: {filename} (Part {i//chunk_size + 1})\nContent: {chunk_text}\n"
            chunks.append(prompt)
        
        return chunks
    
    def load_model_and_tokenizer(self):
        """Load the model and tokenizer with proper configuration."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with proper configuration for training
        try:
            from transformers import BitsAndBytesConfig
            
            # Configure 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            logger.info("Loading model with 4-bit quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            
        except ImportError:
            logger.info("Loading model without quantization")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        
        # Prepare model for k-bit training (essential for gradient computation)
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Enable gradient checkpointing
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Ensure the model is in training mode
        self.model.train()
    
    def create_dataset(self, training_texts: List[str]):
        """Create training dataset."""
        logger.info("Creating dataset...")
        
        # Create dataset from texts
        self.dataset = Dataset.from_dict({"text": training_texts})
        logger.info(f"Dataset created with {len(self.dataset)} examples")
    
    def train(self):
        """Start the training process using SFTTrainer."""
        logger.info("Starting training...")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Training arguments optimized for your setup
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            fp16=self.config.use_fp16,
            optim="paged_adamw_8bit",  # Memory efficient optimizer
            dataloader_pin_memory=False,
            save_total_limit=2,
            load_best_model_at_end=False,
            report_to="none",
            remove_unused_columns=False,
            dataloader_drop_last=True,
            max_grad_norm=1.0,  # Gradient clipping
            disable_tqdm=True,  # Disable default progress bar to use our custom one
        )
        
        # Initialize custom progress callback
        progress_callback = DetailedProgressCallback()
        
        # Try different SFTTrainer initialization approaches for compatibility
        try:
            # First try with tokenizer parameter (newer versions)
            trainer = SFTTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset,
                tokenizer=self.tokenizer,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                packing=False,
                callbacks=[progress_callback],
            )
        except TypeError:
            try:
                # Try without tokenizer parameter (older versions)
                trainer = SFTTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.dataset,
                    dataset_text_field="text",
                    max_seq_length=self.config.max_seq_length,
                    packing=False,
                    callbacks=[progress_callback],
                )
                # Set tokenizer manually
                trainer.tokenizer = self.tokenizer
            except TypeError:
                # Fallback to basic Trainer with data collator
                logger.warning("SFTTrainer not compatible, using basic Trainer")
                
                def tokenize_function(examples):
                    return self.tokenizer(
                        examples["text"],
                        truncation=True,
                        padding="max_length",
                        max_length=self.config.max_seq_length,
                        return_tensors="pt",
                    )
                
                # Tokenize dataset
                tokenized_dataset = self.dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["text"],
                )
                
                # Add labels for language modeling
                def add_labels(examples):
                    examples["labels"] = examples["input_ids"].copy()
                    return examples
                
                tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
                
                # Data collator for language modeling
                from transformers import DataCollatorForLanguageModeling
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False,
                )
                
                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=tokenized_dataset,
                    data_collator=data_collator,
                    callbacks=[progress_callback],
                )
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Print training info
        total_steps = len(trainer.get_train_dataloader()) * self.config.num_epochs // self.config.gradient_accumulation_steps
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Training examples: {len(self.dataset)}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        
        # Train the model
        try:
            trainer.train()
            
            # Save the final model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            logger.info(f"Training completed! Model saved to {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            # Try to save progress if training fails
            try:
                trainer.save_model(f"{self.config.output_dir}_checkpoint")
                logger.info(f"Saved checkpoint to {self.config.output_dir}_checkpoint")
            except:
                pass
            raise
    
    def test_model(self):
        """Test the trained model with a sample prompt."""
        logger.info("Testing trained model...")
        
        try:
            # Load the trained model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            model = PeftModel.from_pretrained(base_model, self.config.output_dir)
            model.eval()
            
            # Test prompt
            test_prompt = "Document: procurement_guidelines.pdf\nContent: What are the key requirements for public procurement?"
            
            inputs = self.tokenizer.encode(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Model response: {response[len(test_prompt):]}")
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")

def main():
    """Main training function."""
    
    # Configuration optimized for RTX 3050 4GB
    config = FineTuningConfig(
        jsonl_file="pdfs_dataset.jsonl",
        output_dir=f"tinyllama_procurement_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        num_epochs=2,  # Reduced epochs for testing
        batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_seq_length=1024,  # Reduced for memory
    )
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 6:
            logger.warning("GPU memory is limited. Using conservative settings.")
            config.batch_size = 1
            config.max_seq_length = 512
    else:
        logger.error("CUDA not available. This script requires a GPU.")
        return
    
    # Initialize trainer
    trainer = TinyLlamaFineTuner(config)
    
    try:
        # Load and prepare data
        pdf_data = trainer.load_jsonl_data()
        training_texts = trainer.prepare_training_data(pdf_data)
        
        if not training_texts:
            logger.error("No training data available!")
            return
        
        # Load model and tokenizer
        trainer.load_model_and_tokenizer()
        
        # Create dataset
        trainer.create_dataset(training_texts)
        
        # Train
        trainer.train()
        
        # Test the model
        trainer.test_model()
        
        logger.info("Fine-tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()