import json
import time
import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

class ModelComparison:
    def __init__(self, gemma_path: str, openai_api_key: str, eval_file: str):
        """
        Initialize the comparison framework
        
        Args:
            gemma_path: Path to fine-tuned Gemma model
            openai_api_key: OpenAI API key
            eval_file: Path to eval_qa_pairs.jsonl
        """
        self.gemma_path = gemma_path
        self.eval_file = eval_file
        
        # Initialize OpenAI client (using gpt-4o-mini - cheapest model)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.gpt_model = "gpt-4o-mini"
        
        # Load evaluation data (JSONL format)
        self.eval_data = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    self.eval_data.append(json.loads(line.strip()))
        
        # Initialize sentence transformer for semantic similarity
        print("Loading sentence transformer for semantic evaluation...")
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load Gemma model
        print(f"Loading fine-tuned Gemma model from {gemma_path}...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        try:
            self.gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_path)
            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                gemma_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading model from {gemma_path}: {e}")
            print("Trying to load base model and adapter separately...")
            
            # Try loading base model first
            base_model_name = "google/gemma-2b-it"  # Fallback to base model
            print(f"Loading base model: {base_model_name}")
            self.gemma_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Force model to GPU if available
            if torch.cuda.is_available():
                self.gemma_model = self.gemma_model.cuda()
                print(f"Model moved to GPU: {next(self.gemma_model.parameters()).device}")
            
            # Try to load adapter
            try:
                from peft import PeftModel
                print(f"Loading PEFT adapter from {gemma_path}")
                self.gemma_model = PeftModel.from_pretrained(self.gemma_model, gemma_path)
                print("PEFT adapter loaded successfully!")
            except Exception as adapter_error:
                print(f"Could not load adapter: {adapter_error}")
                print("Continuing with base model only...")
        
        # Ensure model is in eval mode for inference
        self.gemma_model.eval()
        
        # Check final device placement
        if hasattr(self.gemma_model, 'parameters'):
            device = next(self.gemma_model.parameters()).device
            print(f"Model is on device: {device}")
        
        self.results = []
        
    def get_gemma_response(self, question: str) -> Tuple[str, float]:
        """Get response from fine-tuned Gemma model"""
        prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = self.gemma_tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.gemma_model.generate(
                **inputs,
                max_new_tokens=256,  # Reduced from 512 for speed
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.gemma_tokenizer.eos_token_id,
                use_cache=True,  # Enable KV cache for speed
                early_stopping=True,  # Stop when EOS is generated
            )
        response_time = time.time() - start_time
        
        response = self.gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the model's response
        response = response.split("<start_of_turn>model\n")[-1].strip()
        
        # Clear GPU memory after each generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response, response_time
    
    def get_chatgpt_response(self, question: str) -> Tuple[str, float]:
        """Get response from ChatGPT (gpt-4o-mini)"""
        start_time = time.time()
        try:
            response = self.openai_client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": "You are an expert on Philippine procurement law, specifically RA 9184 and its implementing rules and regulations. Provide accurate, detailed answers about procurement processes."},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=512
            )
            response_time = time.time() - start_time
            return response.choices[0].message.content, response_time
        except Exception as e:
            print(f"Error getting ChatGPT response: {e}")
            return f"Error: {str(e)}", 0.0
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embeddings = self.semantic_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def calculate_metrics(self, generated: str, reference: str) -> Dict:
        """Calculate various metrics for comparison"""
        metrics = {}
        
        # Semantic similarity with reference answer
        metrics['semantic_similarity'] = self.calculate_semantic_similarity(generated, reference)
        
        # Length metrics
        metrics['response_length'] = len(generated.split())
        metrics['reference_length'] = len(reference.split())
        metrics['length_ratio'] = metrics['response_length'] / max(metrics['reference_length'], 1)
        
        # Simple keyword overlap (procurement-specific terms)
        procurement_keywords = [
            'RA 9184', 'procurement', 'bidding', 'contract', 'BAC',
            'evaluation', 'technical', 'financial', 'eligibility',
            'award', 'notice', 'specifications', 'compliance'
        ]
        
        gen_lower = generated.lower()
        ref_lower = reference.lower()
        
        gen_keywords = sum(1 for kw in procurement_keywords if kw.lower() in gen_lower)
        ref_keywords = sum(1 for kw in procurement_keywords if kw.lower() in ref_lower)
        
        metrics['keyword_coverage'] = gen_keywords / max(ref_keywords, 1) if ref_keywords > 0 else 0
        
        return metrics
    
    def run_comparison(self):
        """Run comprehensive comparison"""
        print(f"\n{'='*80}")
        print(f"Starting Comprehensive Model Comparison")
        print(f"{'='*80}")
        print(f"Fine-tuned Model: {self.gemma_path}")
        print(f"ChatGPT Model: {self.gpt_model}")
        print(f"Evaluation Dataset: {len(self.eval_data)} questions")
        print(f"{'='*80}\n")
        
        for idx, qa_pair in enumerate(self.eval_data):
            question = qa_pair['question']
            reference = qa_pair['answer']
            
            print(f"\nProcessing [{idx+1}/{len(self.eval_data)}]: {question[:80]}...")
            
            # Get responses from both models
            gemma_response, gemma_time = self.get_gemma_response(question)
            time.sleep(0.5)  # Brief pause between models
            chatgpt_response, chatgpt_time = self.get_chatgpt_response(question)
            time.sleep(0.5)  # Rate limiting for API
            
            # Calculate metrics for both
            gemma_metrics = self.calculate_metrics(gemma_response, reference)
            chatgpt_metrics = self.calculate_metrics(chatgpt_response, reference)
            
            # Store results
            result = {
                'question_id': idx + 1,
                'question': question,
                'reference_answer': reference,
                'gemma_response': gemma_response,
                'gemma_time': gemma_time,
                'gemma_semantic_similarity': gemma_metrics['semantic_similarity'],
                'gemma_length': gemma_metrics['response_length'],
                'gemma_keyword_coverage': gemma_metrics['keyword_coverage'],
                'chatgpt_response': chatgpt_response,
                'chatgpt_time': chatgpt_time,
                'chatgpt_semantic_similarity': chatgpt_metrics['semantic_similarity'],
                'chatgpt_length': chatgpt_metrics['response_length'],
                'chatgpt_keyword_coverage': chatgpt_metrics['keyword_coverage'],
            }
            
            self.results.append(result)
            
            print(f"  Gemma Similarity: {gemma_metrics['semantic_similarity']:.3f} | Time: {gemma_time:.2f}s")
            print(f"  ChatGPT Similarity: {chatgpt_metrics['semantic_similarity']:.3f} | Time: {chatgpt_time:.2f}s")
        
        print(f"\n{'='*80}")
        print("Comparison Complete!")
        print(f"{'='*80}\n")
    
    def generate_report(self, output_dir: str = "comparison_results"):
        """Generate comprehensive comparison report"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate aggregate statistics
        stats = {
            'gemma': {
                'avg_semantic_similarity': df['gemma_semantic_similarity'].mean(),
                'std_semantic_similarity': df['gemma_semantic_similarity'].std(),
                'avg_response_time': df['gemma_time'].mean(),
                'avg_length': df['gemma_length'].mean(),
                'avg_keyword_coverage': df['gemma_keyword_coverage'].mean(),
            },
            'chatgpt': {
                'avg_semantic_similarity': df['chatgpt_semantic_similarity'].mean(),
                'std_semantic_similarity': df['chatgpt_semantic_similarity'].std(),
                'avg_response_time': df['chatgpt_time'].mean(),
                'avg_length': df['chatgpt_length'].mean(),
                'avg_keyword_coverage': df['chatgpt_keyword_coverage'].mean(),
            }
        }
        
        # Generate visualizations
        self._create_visualizations(df, output_dir, timestamp)
        
        # Save detailed results
        df.to_csv(f"{output_dir}/detailed_results_{timestamp}.csv", index=False)
        
        # Save statistics
        with open(f"{output_dir}/statistics_{timestamp}.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate text report
        report = self._generate_text_report(stats, df)
        with open(f"{output_dir}/report_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_dir}/")
        print(f"{'='*80}\n")
        print(report)
        
        return stats
    
    def _create_visualizations(self, df: pd.DataFrame, output_dir: str, timestamp: str):
        """Create comparison visualizations"""
        sns.set_style("whitegrid")
        
        # 1. Semantic Similarity Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Box plot comparison
        ax = axes[0, 0]
        data_to_plot = [df['gemma_semantic_similarity'], df['chatgpt_semantic_similarity']]
        ax.boxplot(data_to_plot, labels=['Fine-tuned Gemma', 'ChatGPT (GPT-4o-mini)'])
        ax.set_ylabel('Semantic Similarity Score')
        ax.set_title('Semantic Similarity Distribution')
        ax.grid(True, alpha=0.3)
        
        # Response time comparison
        ax = axes[0, 1]
        data_to_plot = [df['gemma_time'], df['chatgpt_time']]
        ax.boxplot(data_to_plot, labels=['Fine-tuned Gemma', 'ChatGPT (GPT-4o-mini)'])
        ax.set_ylabel('Response Time (seconds)')
        ax.set_title('Response Time Distribution')
        ax.grid(True, alpha=0.3)
        
        # Keyword coverage comparison
        ax = axes[1, 0]
        x = np.arange(2)
        means = [df['gemma_keyword_coverage'].mean(), df['chatgpt_keyword_coverage'].mean()]
        stds = [df['gemma_keyword_coverage'].std(), df['chatgpt_keyword_coverage'].std()]
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(['Fine-tuned Gemma', 'ChatGPT (GPT-4o-mini)'])
        ax.set_ylabel('Keyword Coverage Score')
        ax.set_title('Average Keyword Coverage')
        ax.grid(True, alpha=0.3)
        
        # Scatter plot: Similarity vs Response Time
        ax = axes[1, 1]
        ax.scatter(df['gemma_time'], df['gemma_semantic_similarity'], 
                  alpha=0.6, label='Fine-tuned Gemma', s=50)
        ax.scatter(df['chatgpt_time'], df['chatgpt_semantic_similarity'], 
                  alpha=0.6, label='ChatGPT (GPT-4o-mini)', s=50)
        ax.set_xlabel('Response Time (seconds)')
        ax.set_ylabel('Semantic Similarity Score')
        ax.set_title('Similarity vs Response Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/comparison_charts_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Individual question comparison
        fig, ax = plt.subplots(figsize=(15, 8))
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df['gemma_semantic_similarity'], width, 
               label='Fine-tuned Gemma', alpha=0.8)
        ax.bar(x + width/2, df['chatgpt_semantic_similarity'], width, 
               label='ChatGPT (GPT-4o-mini)', alpha=0.8)
        
        ax.set_xlabel('Question ID')
        ax.set_ylabel('Semantic Similarity Score')
        ax.set_title('Per-Question Semantic Similarity Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/per_question_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, stats: Dict, df: pd.DataFrame) -> str:
        """Generate comprehensive text report"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE MODEL COMPARISON REPORT")
        report.append("Philippine Procurement Domain Knowledge Evaluation")
        report.append("=" * 80)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nModels Compared:")
        report.append(f"  1. Fine-tuned Gemma 3 1B (from {self.gemma_path})")
        report.append(f"  2. ChatGPT GPT-4o-mini (OpenAI)")
        report.append(f"\nEvaluation Dataset: {len(self.eval_data)} Q&A pairs")
        
        report.append("\n" + "=" * 80)
        report.append("SUMMARY STATISTICS")
        report.append("=" * 80)
        
        # Semantic Similarity
        report.append("\n1. SEMANTIC SIMILARITY (Higher is Better)")
        report.append("-" * 40)
        report.append(f"Fine-tuned Gemma:")
        report.append(f"  Mean: {stats['gemma']['avg_semantic_similarity']:.4f}")
        report.append(f"  Std:  {stats['gemma']['std_semantic_similarity']:.4f}")
        report.append(f"\nChatGPT (GPT-4o-mini):")
        report.append(f"  Mean: {stats['chatgpt']['avg_semantic_similarity']:.4f}")
        report.append(f"  Std:  {stats['chatgpt']['std_semantic_similarity']:.4f}")
        
        diff = stats['gemma']['avg_semantic_similarity'] - stats['chatgpt']['avg_semantic_similarity']
        winner = "Fine-tuned Gemma" if diff > 0 else "ChatGPT"
        report.append(f"\n→ Winner: {winner} (Δ = {abs(diff):.4f})")
        
        # Response Time
        report.append("\n\n2. RESPONSE TIME (Lower is Better)")
        report.append("-" * 40)
        report.append(f"Fine-tuned Gemma:")
        report.append(f"  Mean: {stats['gemma']['avg_response_time']:.3f}s")
        report.append(f"\nChatGPT (GPT-4o-mini):")
        report.append(f"  Mean: {stats['chatgpt']['avg_response_time']:.3f}s")
        
        time_diff = stats['gemma']['avg_response_time'] - stats['chatgpt']['avg_response_time']
        winner = "Fine-tuned Gemma" if time_diff < 0 else "ChatGPT"
        report.append(f"\n→ Winner: {winner} ({abs(time_diff):.3f}s faster)")
        
        # Keyword Coverage
        report.append("\n\n3. DOMAIN KEYWORD COVERAGE (Higher is Better)")
        report.append("-" * 40)
        report.append(f"Fine-tuned Gemma:")
        report.append(f"  Mean: {stats['gemma']['avg_keyword_coverage']:.4f}")
        report.append(f"\nChatGPT (GPT-4o-mini):")
        report.append(f"  Mean: {stats['chatgpt']['avg_keyword_coverage']:.4f}")
        
        kw_diff = stats['gemma']['avg_keyword_coverage'] - stats['chatgpt']['avg_keyword_coverage']
        winner = "Fine-tuned Gemma" if kw_diff > 0 else "ChatGPT"
        report.append(f"\n→ Winner: {winner} (Δ = {abs(kw_diff):.4f})")
        
        # Response Length
        report.append("\n\n4. AVERAGE RESPONSE LENGTH")
        report.append("-" * 40)
        report.append(f"Fine-tuned Gemma: {stats['gemma']['avg_length']:.1f} words")
        report.append(f"ChatGPT (GPT-4o-mini): {stats['chatgpt']['avg_length']:.1f} words")
        
        # Overall Winner
        report.append("\n\n" + "=" * 80)
        report.append("OVERALL ASSESSMENT")
        report.append("=" * 80)
        
        gemma_wins = 0
        chatgpt_wins = 0
        
        if stats['gemma']['avg_semantic_similarity'] > stats['chatgpt']['avg_semantic_similarity']:
            gemma_wins += 1
        else:
            chatgpt_wins += 1
            
        if stats['gemma']['avg_response_time'] < stats['chatgpt']['avg_response_time']:
            gemma_wins += 1
        else:
            chatgpt_wins += 1
            
        if stats['gemma']['avg_keyword_coverage'] > stats['chatgpt']['avg_keyword_coverage']:
            gemma_wins += 1
        else:
            chatgpt_wins += 1
        
        report.append(f"\nMetric Wins:")
        report.append(f"  Fine-tuned Gemma: {gemma_wins}/3")
        report.append(f"  ChatGPT (GPT-4o-mini): {chatgpt_wins}/3")
        
        # Detailed Analysis
        report.append("\n\nDETAILED ANALYSIS:")
        report.append("-" * 40)
        
        if stats['gemma']['avg_semantic_similarity'] > stats['chatgpt']['avg_semantic_similarity']:
            report.append("\n✓ Fine-tuned Gemma shows better semantic alignment with reference")
            report.append("  answers, suggesting better domain-specific knowledge capture.")
        else:
            report.append("\n✓ ChatGPT shows better semantic alignment, indicating strong")
            report.append("  general-purpose capabilities on this domain.")
        
        if stats['gemma']['avg_response_time'] < stats['chatgpt']['avg_response_time']:
            report.append("\n✓ Fine-tuned Gemma is faster, beneficial for real-time applications.")
        else:
            report.append("\n✓ ChatGPT API provides faster responses.")
        
        if stats['gemma']['avg_keyword_coverage'] > stats['chatgpt']['avg_keyword_coverage']:
            report.append("\n✓ Fine-tuned Gemma demonstrates better domain terminology usage.")
        else:
            report.append("\n✓ ChatGPT demonstrates better domain terminology usage.")
        
        report.append("\n\n" + "=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)
        report.append("\nBased on the evaluation results:")
        
        if gemma_wins > chatgpt_wins:
            report.append("\n→ The fine-tuned Gemma model shows superior performance for")
            report.append("  Philippine procurement domain tasks. Consider deploying it for")
            report.append("  production use, especially if data privacy and local hosting")
            report.append("  are priorities.")
        else:
            report.append("\n→ ChatGPT (GPT-4o-mini) shows competitive performance. The fine-tuned")
            report.append("  model may benefit from additional training data or hyperparameter")
            report.append("  tuning to improve domain-specific capabilities.")
        
        report.append("\n\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main execution function"""
    # Configuration
    GEMMA_MODEL_PATH = "gemma3_1b_procurement_weska_v3"
    EVAL_FILE = "eval_qa_pairs.jsonl"  # Changed to .jsonl format
    
    # Option 1: Put your API key directly here
    OPENAI_API_KEY = "REPLACED_FOR_SECURITY"
    
    # Option 2: Or use environment variable (comment out line above and uncomment below)
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("Error: Please set OPENAI_API_KEY in the code or as environment variable")
        return
    
    # Initialize comparison
    comparison = ModelComparison(
        gemma_path=GEMMA_MODEL_PATH,
        openai_api_key=OPENAI_API_KEY,
        eval_file=EVAL_FILE
    )
    
    # Run comparison
    comparison.run_comparison()
    
    # Generate report
    stats = comparison.generate_report()
    
    print("\n✓ Comparison complete! Check the 'comparison_results' folder for detailed outputs.")


if __name__ == "__main__":
    main()