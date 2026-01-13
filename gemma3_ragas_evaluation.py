#!/usr/bin/env python3
"""
RAGAS Evaluation for Gemma3 1B Fine-tuned Procurement Model
This script evaluates the Gemma3 1B fine-tuned model using RAGAS metrics on procurement Q&A pairs.
"""

import os
import json
import pandas as pd
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import warnings
import time
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# ================= CONFIG =================
GEMMA_MODEL_PATH = "gemma3_1b_procurement_weska_v2"
QA_DATA_PATH = "data/eval_qa_pairs.jsonl"
CHROMA_DB_PATH = "./procure_chroma_db"
COLLECTION_NAME = "procurement_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy, 
        context_precision,
        context_relevancy,
        context_recall,
        answer_correctness,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    print("✅ RAGAS library loaded successfully")
except ImportError as e:
    print(f"❌ RAGAS not available: {e}")
    print("Install with: pip install ragas")
    RAGAS_AVAILABLE = False

class GemmaRAGEvaluator:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.qa_data = []
        self.results = []
        
    def setup_models(self):
        """Initialize all models and components"""
        print("\n🚀 Setting up models...")
        
        # 1. Load Gemma fine-tuned model
        print(f"📚 Loading Gemma model from {GEMMA_MODEL_PATH}...")
        self.tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_PATH)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "right"
            
        self.model = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Gemma model loaded")
        
        # 2. Load embedding model for retrieval
        print(f"🔍 Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ Embedding model loaded")
        
        # 3. Connect to ChromaDB
        print(f"🗄️ Connecting to ChromaDB: {CHROMA_DB_PATH}")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
            doc_count = self.collection.count()
            print(f"✅ Connected to collection '{COLLECTION_NAME}' with {doc_count} documents")
        except Exception as e:
            print(f"❌ Error connecting to ChromaDB collection: {e}")
            print("Make sure the ChromaDB collection exists and is populated")
            return False
            
        return True
        
    def load_qa_data(self):
        """Load Q&A evaluation data"""
        print(f"\n📖 Loading Q&A data from {QA_DATA_PATH}")
        try:
            with open(QA_DATA_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    self.qa_data.append(data)
            print(f"✅ Loaded {len(self.qa_data)} Q&A pairs")
            return True
        except Exception as e:
            print(f"❌ Error loading Q&A data: {e}")
            return False
            
    def retrieve_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant context using ChromaDB"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            if results and results['documents'] and results['documents'][0]:
                contexts = results['documents'][0]
                return contexts
            else:
                return []
        except Exception as e:
            print(f"⚠️ Error retrieving context: {e}")
            return []
            
    def generate_answer(self, question: str, context: str = "") -> str:
        """Generate answer using fine-tuned Gemma model"""
        try:
            # Create prompt with context if available
            if context:
                prompt = f"""Based on the following context about Philippine procurement law, answer the question accurately and concisely.

Context: {context}

Question: {question}

Answer:"""
            else:
                prompt = f"""Answer the following question about Philippine procurement law accurately and concisely.

Question: {question}

Answer:"""
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the new generated text
            answer = full_response[len(prompt):].strip()
            return answer
            
        except Exception as e:
            print(f"⚠️ Error generating answer: {e}")
            return "Error generating response"
            
    def run_evaluation(self, max_samples: int = 50):
        """Run RAGAS evaluation on Q&A pairs"""
        if not RAGAS_AVAILABLE:
            print("❌ RAGAS not available. Cannot run evaluation.")
            return False
            
        print(f"\n🎯 Starting RAGAS evaluation with {min(max_samples, len(self.qa_data))} samples...")
        
        # Prepare data for evaluation
        evaluation_data = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': []
        }
        
        # Process each Q&A pair
        samples_to_process = min(max_samples, len(self.qa_data))
        for i, qa_pair in enumerate(self.qa_data[:samples_to_process]):
            print(f"\n📝 Processing sample {i+1}/{samples_to_process}")
            
            question = qa_pair['question']
            ground_truth = qa_pair['answer']
            
            print(f"Q: {question}")
            
            # Retrieve relevant context
            print("   🔍 Retrieving context...")
            contexts = self.retrieve_context(question, k=5)
            context_text = " ".join(contexts) if contexts else ""
            
            # Generate answer
            print("   🤖 Generating answer...")
            generated_answer = self.generate_answer(question, context_text)
            
            print(f"   A: {generated_answer[:100]}...")
            
            # Store for RAGAS evaluation
            evaluation_data['question'].append(question)
            evaluation_data['answer'].append(generated_answer)
            evaluation_data['contexts'].append(contexts)
            evaluation_data['ground_truth'].append(ground_truth)
            
            # Add delay to prevent overheating
            if i % 10 == 0 and i > 0:
                print("   ⏸️ Cooling down...")
                time.sleep(2)
                
        # Convert to Dataset for RAGAS
        print("\n📊 Running RAGAS evaluation...")
        dataset = Dataset.from_dict(evaluation_data)
        
        # Define metrics to evaluate
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_relevancy, 
            answer_correctness,
            answer_similarity
        ]
        
        try:
            # Run RAGAS evaluation
            results = evaluate(
                dataset=dataset,
                metrics=metrics
            )
            
            self.results = results
            return True
            
        except Exception as e:
            print(f"❌ Error during RAGAS evaluation: {e}")
            return False
            
    def print_results(self):
        """Print evaluation results"""
        if not self.results:
            print("No results to display")
            return
            
        print(f"\n{'='*70}")
        print(f"🎯 GEMMA3 1B RAGAS EVALUATION RESULTS")
        print(f"{'='*70}")
        
        print(f"\nModel: Gemma3 1B Fine-tuned (procurement)")
        print(f"Evaluation Dataset: {len(self.qa_data)} Q&A pairs")
        print(f"ChromaDB Collection: {COLLECTION_NAME}")
        
        print(f"\n📊 RAGAS METRICS:")
        print(f"{'Metric':<20} {'Score':<10} {'Status'}")
        print("-" * 40)
        
        # Define target thresholds
        targets = {
            'faithfulness': 0.80,
            'answer_relevancy': 0.70,
            'context_precision': 0.70,
            'context_relevancy': 0.60,
            'answer_correctness': 0.70,
            'answer_similarity': 0.75
        }
        
        for metric_name, score in self.results.items():
            if isinstance(score, (int, float)):
                target = targets.get(metric_name, 0.70)
                status = "✅" if score >= target else "❌"
                print(f"{metric_name:<20} {score:<10.4f} {status}")
        
        # Calculate overall score
        metric_scores = [v for k, v in self.results.items() if isinstance(v, (int, float))]
        if metric_scores:
            overall_score = sum(metric_scores) / len(metric_scores)
            print(f"\n🎯 Overall Score: {overall_score:.4f}")
            
            if overall_score >= 0.75:
                print("🏆 EXCELLENT performance!")
            elif overall_score >= 0.65:
                print("✅ GOOD performance")
            elif overall_score >= 0.55:
                print("⚠️ FAIR performance - needs improvement")
            else:
                print("❌ POOR performance - significant improvement needed")
                
    def save_results(self, output_dir: str = "results"):
        """Save detailed results to files"""
        if not self.results:
            print("No results to save")
            return
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to JSON
        json_path = f"{output_dir}/gemma3_ragas_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Convert results to serializable format
            serializable_results = {}
            for k, v in self.results.items():
                if isinstance(v, (int, float, str, bool)):
                    serializable_results[k] = v
                else:
                    serializable_results[k] = str(v)
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to: {json_path}")
        
        # Save summary to CSV
        csv_data = []
        for metric_name, score in self.results.items():
            if isinstance(score, (int, float)):
                csv_data.append({
                    'metric': metric_name,
                    'score': score,
                    'model': 'Gemma3-1B-Procurement'
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = f"{output_dir}/gemma3_ragas_summary_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"💾 Summary saved to: {csv_path}")
            
def main():
    """Main evaluation function"""
    print("🚀 GEMMA3 1B RAGAS EVALUATION")
    print("=" * 50)
    
    # Check if RAGAS is available
    if not RAGAS_AVAILABLE:
        print("❌ RAGAS library is required but not installed.")
        print("Install with: pip install ragas")
        return
        
    # Initialize evaluator
    evaluator = GemmaRAGEvaluator()
    
    # Setup models
    if not evaluator.setup_models():
        print("❌ Failed to setup models")
        return
        
    # Load Q&A data
    if not evaluator.load_qa_data():
        print("❌ Failed to load Q&A data")
        return
        
    # Run evaluation
    max_samples = int(input(f"\nHow many samples to evaluate? (max {len(evaluator.qa_data)}, default 30): ").strip() or "30")
    
    if evaluator.run_evaluation(max_samples):
        # Print and save results
        evaluator.print_results()
        evaluator.save_results()
        print("\n✅ Evaluation completed successfully!")
    else:
        print("❌ Evaluation failed")

if __name__ == "__main__":
    main()