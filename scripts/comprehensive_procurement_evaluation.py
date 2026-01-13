#!/usr/bin/env python3
"""
Comprehensive Evaluation: Google Gemini vs Fine-tuned Gemma v2 vs RAG + Gemma v2
Philippine Government Procurement Law (RA 9184) Q&A System Comparison
"""

import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# Core libraries
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import faiss

# Evaluation metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download("punkt", quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except:
    BERTSCORE_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# ================= CONFIGURATION =================
CONFIG = {
    # Models
    "base_model": "google/gemma-3-1b-it",
    "finetuned_model": "./gemma3_1b_procurement_weska_v2",
    
    # Dataset
    "evaluation_dataset": "eval_qa_pairs.jsonl",  # 100 test Q&A pairs
    "documents_corpus": "documents.jsonl",       # For RAG retrieval
    "embeddings_file": "results/all-MiniLM-L6-v2_embeddings_results.json",
    
    # Evaluation settings
    "num_eval_samples": 100,
    "random_seed": 42,
    "max_new_tokens": 200,
    "temperature": 0.1,  # Low for factual consistency
    
    # RAG settings
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k_retrieval": 5,
    "max_context_length": 1500,
    
    # Google Gemini settings
    "gemini_model": "gemini-2.5-flash",  # Updated to working model
    "gemini_api_key": "AIzaSyDsD1SI5G83btLrR_wRgmcaCU3jYFzZ2yQ",  # Set your API key
    
    # Output
    "results_dir": "evaluation_results",
    "timestamp": time.strftime("%Y%m%d_%H%M%S")
}

# System prompt for all models
SYSTEM_PROMPT = """You are an expert on Philippine Government Procurement (RA 9184 and its IRR). Answer factually and concisely, and cite sections or rules when applicable."""

class ProcurementEvaluator:
    def __init__(self, config):
        self.config = config
        self.results_dir = Path(config["results_dir"])
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.embedding_model = None
        self.vector_index = None
        self.documents = []
        self.gemma_model = None
        self.gemma_tokenizer = None
        
        # Results storage
        self.responses = []
        self.metrics = defaultdict(list)
        
        print(f"🔧 Initializing Procurement Law Evaluation Framework")
        print(f"📊 Evaluating {config['num_eval_samples']} Q&A pairs")
        
    def setup_models(self):
        """Initialize all models and components"""
        print("\n🚀 Setting up models...")
        
        # 1. Setup embedding model and RAG
        self._setup_rag_system()
        
        # 2. Setup Gemma model
        self._setup_gemma_model()
        
        # 3. Verify Gemini setup
        self._verify_gemini_setup()
        
        print("✅ All models initialized successfully!")
        
    def _setup_rag_system(self):
        """Setup RAG retrieval system"""
        print("📚 Setting up RAG system...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(self.config["embedding_model"])
        
        # Load documents
        documents_file = Path(self.config["documents_corpus"])
        if documents_file.exists():
            with open(documents_file, 'r', encoding='utf-8') as f:
                self.documents = [json.loads(line) for line in f]
            print(f"   Loaded {len(self.documents)} documents")
        else:
            print(f"   ⚠️  Documents file not found: {documents_file}")
            self.documents = []
            return
        
        # Load or create embeddings
        embeddings_file = Path(self.config["embeddings_file"])
        
        # Check if this is a results file or actual embeddings
        embeddings_available = False
        if embeddings_file.exists():
            try:
                with open(embeddings_file, 'r') as f:
                    embeddings_data = json.load(f)
                
                # Check if this contains actual embeddings (dict with numeric values)
                # vs results (dict with metadata like "model", "total_queries")
                if isinstance(embeddings_data, dict):
                    # Look for metadata keys that indicate this is a results file
                    metadata_keys = {'model', 'total_queries', 'avg_query_time_ms', 'recall@1', 'recall@3', 'recall@5', 'recall@10'}
                    if any(key in embeddings_data for key in metadata_keys):
                        print(f"   File contains evaluation results metadata, not embeddings")
                        embeddings_available = False
                    else:
                        # Check if values are lists of numbers (embeddings)
                        sample_key = next(iter(embeddings_data.keys()))
                        sample_value = embeddings_data[sample_key]
                        
                        if isinstance(sample_value, list) and len(sample_value) > 10 and all(isinstance(x, (int, float)) for x in sample_value[:5]):
                            # This looks like actual embeddings
                            embeddings_available = True
                            print(f"   Found precomputed embeddings")
                        else:
                            print(f"   File contains data but not embeddings")
                            embeddings_available = False
                else:
                    print(f"   File format not recognized")
                    embeddings_available = False
                        
            except Exception as e:
                print(f"   Error reading embeddings file: {e}")
                embeddings_available = False
        
        if embeddings_available:
            # Use precomputed embeddings
            doc_embeddings = []
            self.doc_ids = []
            
            for doc_id, embedding in embeddings_data.items():
                doc_embeddings.append(embedding)
                self.doc_ids.append(doc_id)
            
            embeddings_matrix = np.array(doc_embeddings, dtype=np.float32)
            
            # Build FAISS index
            dimension = embeddings_matrix.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            faiss.normalize_L2(embeddings_matrix)  # Normalize for cosine similarity
            self.vector_index.add(embeddings_matrix)
            
            print(f"   Built FAISS index with {len(doc_embeddings)} precomputed embeddings")
            
        else:
            # Generate embeddings on the fly
            print(f"   Generating embeddings for {len(self.documents)} documents...")
            print("   This may take a few minutes...")
            
            doc_texts = []
            self.doc_ids = []
            
            for i, doc in enumerate(self.documents):
                doc_text = doc.get("text", "")
                if doc_text.strip():  # Only include non-empty documents
                    doc_texts.append(doc_text)
                    self.doc_ids.append(str(i))
            
            if doc_texts:
                # Generate embeddings in batches to avoid memory issues
                batch_size = 32
                doc_embeddings = []
                
                for i in range(0, len(doc_texts), batch_size):
                    batch = doc_texts[i:i+batch_size]
                    batch_embeddings = self.embedding_model.encode(batch, show_progress_bar=True)
                    doc_embeddings.extend(batch_embeddings.tolist())
                
                embeddings_matrix = np.array(doc_embeddings, dtype=np.float32)
                
                # Build FAISS index
                dimension = embeddings_matrix.shape[1]
                self.vector_index = faiss.IndexFlatIP(dimension)
                faiss.normalize_L2(embeddings_matrix)
                self.vector_index.add(embeddings_matrix)
                
                print(f"   Built FAISS index with {len(doc_embeddings)} generated embeddings")
                
                # Optionally save embeddings for future use
                embeddings_dict = {str(i): emb for i, emb in enumerate(doc_embeddings)}
                embeddings_save_file = self.results_dir / "generated_embeddings.json"
                with open(embeddings_save_file, 'w') as f:
                    json.dump(embeddings_dict, f)
                print(f"   Saved embeddings to: {embeddings_save_file}")
                
            else:
                print("   ⚠️  No valid documents found for embedding")
                self.vector_index = None
    
    def _setup_gemma_model(self):
        """Setup fine-tuned Gemma model with improved error handling"""
        print("🤖 Loading fine-tuned Gemma model...")
        
        try:
            # Load tokenizer first
            self.gemma_tokenizer = AutoTokenizer.from_pretrained(
                self.config["base_model"],
                clean_up_tokenization_spaces=True
            )
            
            # Ensure pad token is set
            if self.gemma_tokenizer.pad_token is None:
                self.gemma_tokenizer.pad_token = self.gemma_tokenizer.eos_token
            
            # Try loading base model with different configurations
            base_model = None
            model_configs = [
                # Configuration 1: Most conservative
                {"torch_dtype": torch.float16, "device_map": "auto", "low_cpu_mem_usage": True},
                # Configuration 2: CPU only
                {"torch_dtype": torch.float32, "device_map": "cpu"},
                # Configuration 3: Minimal config
                {"device_map": "auto"}
            ]
            
            for i, config in enumerate(model_configs, 1):
                try:
                    print(f"   Trying model configuration {i}...")
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.config["base_model"],
                        trust_remote_code=True,
                        **config
                    )
                    print(f"   ✅ Base model loaded with configuration {i}")
                    break
                except Exception as e:
                    print(f"   ⚠️  Configuration {i} failed: {e}")
                    continue
            
            if base_model is None:
                raise Exception("All model configurations failed")
            
            # Try loading fine-tuned adapter
            finetuned_dir = Path(self.config["finetuned_model"])
            if finetuned_dir.exists():
                try:
                    print("   Loading PEFT adapter...")
                    self.gemma_model = PeftModel.from_pretrained(
                        base_model, 
                        str(finetuned_dir),
                        torch_dtype=torch.float16,
                        is_trainable=False
                    )
                    print("   ✅ Fine-tuned Gemma model with PEFT adapter loaded successfully")
                except Exception as peft_error:
                    print(f"   ⚠️  PEFT adapter error: {peft_error}")
                    print("   📋 Using base model without fine-tuning")
                    self.gemma_model = base_model
            else:
                print(f"   ⚠️  Fine-tuned model directory not found: {finetuned_dir}")
                print("   📋 Using base model only")
                self.gemma_model = base_model
                
        except Exception as e:
            print(f"   ❌ Error loading Gemma model: {e}")
            print("   📋 Continuing without Gemma model - only Gemini will be evaluated")
            self.gemma_model = None
            self.gemma_tokenizer = None
    
    def _verify_gemini_setup(self):
        """Verify Google Gemini API setup"""
        if not GEMINI_AVAILABLE:
            print("   ⚠️  Google Generative AI library not available. Install: pip install google-generativeai")
            return
        
        if not self.config["gemini_api_key"]:
            print("   ⚠️  Gemini API key not set. Please set CONFIG['gemini_api_key']")
            return
        
        genai.configure(api_key=self.config["gemini_api_key"])
        print("   ✅ Gemini API configured")
    
    def load_evaluation_dataset(self):
        """Load evaluation Q&A pairs"""
        eval_file = Path(self.config["evaluation_dataset"])
        
        if not eval_file.exists():
            print(f"❌ Evaluation dataset not found: {eval_file}")
            print("Please create eval_qa_pairs.jsonl with format:")
            print('{"question": "...", "answer": "..."}')
            return []
        
        qa_pairs = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        qa_pair = json.loads(line)
                        if 'question' in qa_pair and 'answer' in qa_pair:
                            qa_pairs.append(qa_pair)
                        else:
                            print(f"⚠️  Line {line_num}: Missing 'question' or 'answer' field")
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Line {line_num}: JSON error - {e}")
        
        # Sample random subset if needed
        if len(qa_pairs) > self.config["num_eval_samples"]:
            random.seed(self.config["random_seed"])
            qa_pairs = random.sample(qa_pairs, self.config["num_eval_samples"])
        
        print(f"📋 Loaded {len(qa_pairs)} evaluation Q&A pairs")
        return qa_pairs
    
    def retrieve_context(self, question: str) -> str:
        """Retrieve relevant context for RAG"""
        if not self.vector_index or not self.embedding_model:
            return ""
        
        try:
            # Embed question
            question_embedding = self.embedding_model.encode([question])
            question_embedding = question_embedding.astype(np.float32)
            faiss.normalize_L2(question_embedding)
            
            # Search for similar documents
            scores, indices = self.vector_index.search(
                question_embedding, 
                self.config["top_k_retrieval"]
            )
            
            # Collect relevant documents
            context_parts = []
            total_length = 0
            max_length = self.config["max_context_length"]
            
            for i, doc_idx in enumerate(indices[0]):
                if doc_idx < len(self.documents):
                    doc = self.documents[doc_idx]
                    doc_text = doc.get("text", "")
                    
                    if total_length + len(doc_text) <= max_length:
                        context_parts.append(f"Document {i+1}: {doc_text}")
                        total_length += len(doc_text)
                    else:
                        # Truncate last document to fit
                        remaining = max_length - total_length
                        if remaining > 100:  # Only add if meaningful length
                            truncated = doc_text[:remaining] + "..."
                            context_parts.append(f"Document {i+1}: {truncated}")
                        break
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"⚠️  Error in retrieval: {e}")
            return ""
    
    def format_prompt(self, question: str, context: str = "") -> str:
        """Format prompt for consistent evaluation"""
        if context:
            return f"""Context: {context}

Q: {question}
A:"""
        else:
            return f"""Q: {question}
A:"""
    
    def query_gemini(self, question: str) -> str:
        """Query Google Gemini with the question"""
        if not GEMINI_AVAILABLE or not self.config["gemini_api_key"]:
            return "Gemini not available (API key not set)"
        
        try:
            prompt = self.format_prompt(question)
            
            # Initialize the model
            model = genai.GenerativeModel(self.config["gemini_model"])
            
            # Create the full prompt with system context
            full_prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {prompt}"
            
            # Generate response with timeout
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.config["max_new_tokens"],
                    temperature=self.config["temperature"]
                ),
                request_options={
                    "timeout": 30  # 30 second timeout
                }
            )
            
            return response.text.strip() if response.text else "No response generated"
            
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return "Gemini API timeout error"
            elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
                return "Gemini API quota/rate limit error"
            elif "404" in error_msg:
                return f"Gemini model not found: {self.config['gemini_model']}"
            else:
                return f"Gemini API error: {error_msg[:100]}"
    
    def query_gemma(self, question: str, use_rag: bool = False) -> str:
        """Query fine-tuned Gemma model"""
        if not self.gemma_model or not self.gemma_tokenizer:
            return "Gemma model not available"
        
        try:
            # Get context if RAG is enabled
            context = ""
            if use_rag:
                context = self.retrieve_context(question)
            
            # Format prompt
            prompt = self.format_prompt(question, context)
            
            # Apply Gemma chat template
            chat = [
                {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt}"}
            ]
            
            formatted_prompt = self.gemma_tokenizer.apply_chat_template(
                chat, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.gemma_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            if hasattr(self.gemma_model, 'device'):
                inputs = {k: v.to(self.gemma_model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.gemma_model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_new_tokens"],
                    temperature=self.config["temperature"],
                    do_sample=True if self.config["temperature"] > 0 else False,
                    pad_token_id=self.gemma_tokenizer.eos_token_id,
                    eos_token_id=self.gemma_tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.gemma_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Error querying Gemma: {str(e)}"
    
    def calculate_metrics(self, reference: str, generated: str) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        metrics = {}
        
        # BLEU Score
        if NLTK_AVAILABLE:
            try:
                smoothie = SmoothingFunction().method4
                bleu = sentence_bleu(
                    [reference.split()], 
                    generated.split(), 
                    smoothing_function=smoothie
                )
                metrics['bleu'] = bleu
            except:
                metrics['bleu'] = 0.0
        else:
            metrics['bleu'] = 0.0
        
        # ROUGE Scores
        if ROUGE_AVAILABLE:
            try:
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                rouge_scores = scorer.score(reference, generated)
                metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
                metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
                metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
            except:
                metrics['rouge1'] = metrics['rouge2'] = metrics['rougeL'] = 0.0
        else:
            metrics['rouge1'] = metrics['rouge2'] = metrics['rougeL'] = 0.0
        
        # F1 Score (token-level)
        ref_tokens = set(reference.lower().split())
        gen_tokens = set(generated.lower().split())
        
        if len(gen_tokens) == 0:
            metrics['f1'] = 0.0
        else:
            precision = len(ref_tokens.intersection(gen_tokens)) / len(gen_tokens)
            recall = len(ref_tokens.intersection(gen_tokens)) / len(ref_tokens) if len(ref_tokens) > 0 else 0
            metrics['f1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Exact Match
        metrics['exact_match'] = 1.0 if reference.strip().lower() == generated.strip().lower() else 0.0
        
        # BERTScore (if available)
        if BERTSCORE_AVAILABLE:
            try:
                P, R, F1 = bert_score([generated], [reference], lang="en", verbose=False)
                metrics['bert_score'] = F1.item()
            except:
                metrics['bert_score'] = 0.0
        else:
            metrics['bert_score'] = 0.0
        
        return metrics
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n🔄 Starting evaluation...")
        
        # Load evaluation dataset
        qa_pairs = self.load_evaluation_dataset()
        if not qa_pairs:
            return
        
        # Initialize progress tracking
        total_questions = len(qa_pairs)
        
        for i, qa_pair in enumerate(qa_pairs, 1):
            question = qa_pair["question"]
            reference_answer = qa_pair["answer"]
            
            print(f"\n📝 Question {i}/{total_questions}")
            print(f"Q: {question[:80]}...")
            
            # Query all three systems
            responses = {}
            
            # 1. Google Gemini
            print("   💎 Querying Gemini...")
            responses['gemini'] = self.query_gemini(question)
            
            # 2. Fine-tuned Gemma
            print("   🔧 Querying Fine-tuned Gemma...")
            responses['gemma_finetuned'] = self.query_gemma(question, use_rag=False)
            
            # 3. RAG + Fine-tuned Gemma
            print("   📚 Querying RAG + Fine-tuned Gemma...")
            responses['gemma_rag'] = self.query_gemma(question, use_rag=True)
            
            # Calculate metrics for each system
            question_metrics = {}
            for system, response in responses.items():
                question_metrics[system] = self.calculate_metrics(reference_answer, response)
            
            # Store results
            result_entry = {
                "question_id": i,
                "question": question,
                "reference_answer": reference_answer,
                "responses": responses,
                "metrics": question_metrics
            }
            
            self.responses.append(result_entry)
            
            # Update running metrics
            for system in responses.keys():
                for metric_name, metric_value in question_metrics[system].items():
                    self.metrics[f"{system}_{metric_name}"].append(metric_value)
        
        print("\n✅ Evaluation completed!")
        
    def save_results(self):
        """Save detailed results and metrics"""
        timestamp = self.config["timestamp"]
        
        # 1. Save detailed responses
        responses_file = self.results_dir / f"detailed_responses_{timestamp}.json"
        with open(responses_file, 'w', encoding='utf-8') as f:
            json.dump(self.responses, f, indent=2, ensure_ascii=False)
        print(f"💾 Detailed responses saved: {responses_file}")
        
        # 2. Create metrics summary
        summary_metrics = {}
        for metric_key, values in self.metrics.items():
            summary_metrics[metric_key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "count": len(values)
            }
        
        metrics_file = self.results_dir / f"metrics_summary_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(summary_metrics, f, indent=2)
        print(f"📊 Metrics summary saved: {metrics_file}")
        
        # 3. Create comparison table
        self.create_comparison_table(timestamp)
        
    def create_comparison_table(self, timestamp: str):
        """Create comparison table in multiple formats"""
        
        # Extract metrics for table
        systems = ['gemini', 'gemma_finetuned', 'gemma_rag']
        metrics_names = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'f1', 'bert_score', 'exact_match']
        
        # Create DataFrame with proper display names
        system_display_names = {
            'gemini': 'Google Gemini',
            'gemma_finetuned': 'Gemma Finetuned', 
            'gemma_rag': 'Gemma RAG'
        }
        
        table_data = []
        for system in systems:
            row = {'System': system_display_names.get(system, system.replace('_', ' ').title())}
            for metric in metrics_names:
                metric_key = f"{system}_{metric}"
                if metric_key in self.metrics:
                    mean_value = np.mean(self.metrics[metric_key])
                    if metric == 'exact_match':
                        row[metric.upper()] = f"{mean_value:.1%}"
                    else:
                        row[metric.upper()] = f"{mean_value:.3f}"
                else:
                    row[metric.upper()] = "N/A"
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        csv_file = self.results_dir / f"comparison_table_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"📊 Comparison table saved: {csv_file}")
        
        # Print to console
        print("\n" + "="*80)
        print("📊 EVALUATION RESULTS SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Create Markdown table
        md_file = self.results_dir / f"comparison_table_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write("# Philippine Procurement Law Q&A Evaluation Results\n\n")
            f.write(f"**Evaluation Date**: {timestamp}\n")
            f.write(f"**Questions Evaluated**: {self.config['num_eval_samples']}\n")
            f.write(f"**Models Compared**: Google Gemini, Fine-tuned Gemma v2, RAG + Fine-tuned Gemma v2\n\n")
            f.write("## Results Summary\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n## Metric Definitions\n\n")
            f.write("- **BLEU**: N-gram precision overlap (0-1, higher better)\n")
            f.write("- **ROUGE-1/2/L**: Content overlap metrics (0-1, higher better)\n")
            f.write("- **F1**: Harmonic mean of precision/recall (0-1, higher better)\n")
            f.write("- **BERT_SCORE**: Semantic similarity using BERT (0-1, higher better)\n")
            f.write("- **EXACT_MATCH**: Percentage of perfect matches (0-100%)\n")
        
        print(f"📝 Markdown report saved: {md_file}")

def main():
    """Main evaluation execution"""
    print("🚀 Philippine Procurement Law Q&A System Evaluation")
    print("Comparing: Google Gemini vs Fine-tuned Gemma v2 vs RAG + Fine-tuned Gemma v2")
    print("="*80)
    
    # Initialize evaluator
    evaluator = ProcurementEvaluator(CONFIG)
    
    # Setup all models
    evaluator.setup_models()
    
    # Run evaluation
    evaluator.run_evaluation()
    
    # Save and display results
    evaluator.save_results()
    
    print("\n🎉 Evaluation completed successfully!")
    print(f"📁 Results saved in: {evaluator.results_dir}")

if __name__ == "__main__":
    main()