import json
import time
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Set
import pandas as pd
from datetime import datetime

# Suppress all warnings before importing transformers
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import logging
logging.getLogger().setLevel(logging.ERROR)

# Set specific logging levels to reduce warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
logging.getLogger("accelerate.utils.modeling").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.utils.hub").setLevel(logging.ERROR)
logging.getLogger("transformers.utils.generic").setLevel(logging.ERROR)
logging.getLogger("accelerate.accelerator").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("bert_score").setLevel(logging.ERROR)
logging.getLogger("transformers.utils.hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# Toxicity detection
try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    print("Warning: detoxify not installed. Toxicity metrics will be skipped.")
    DETOXIFY_AVAILABLE = False

class AdvancedModelComparison:
    def __init__(self, gemma_path: str, openai_api_key: str, eval_file: str):
        """
        Initialize advanced comparison framework with comprehensive metrics
        
        Args:
            gemma_path: Path to fine-tuned Gemma model
            openai_api_key: OpenAI API key
            eval_file: Path to eval_qa_pairs.jsonl
        """
        self.gemma_path = gemma_path
        self.eval_file = eval_file
        
        # Initialize OpenAI client (gpt-4o-mini - cheapest model)
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.gpt_model = "gpt-4o-mini"
        
        # Load evaluation data (JSONL format)
        self.eval_data = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    self.eval_data.append(json.loads(line.strip()))
        
        print("Initializing evaluation models...")
        
        # Initialize sentence transformer for semantic similarity with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize toxicity detector
        if DETOXIFY_AVAILABLE:
            self.toxicity_model = Detoxify('original')
        else:
            self.toxicity_model = None
        
        # Load Gemma model with warning suppression
        print(f"Loading fine-tuned Gemma model from {gemma_path}...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Suppress warnings during model loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        # Load Gemma model with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gemma_tokenizer = AutoTokenizer.from_pretrained(
                gemma_path,
                clean_up_tokenization_spaces=False  # Suppress tokenization warning
            )
            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                gemma_path,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )        # Ensure model is in eval mode for inference
        self.gemma_model.eval()
        
        # Check final device placement
        if hasattr(self.gemma_model, 'parameters'):
            device = next(self.gemma_model.parameters()).device
            print(f"Model is on device: {device}")
        
        # Clear any initial GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.results = []
        
    def get_gemma_response(self, question: str) -> Tuple[str, float, float]:
        """Get response from fine-tuned Gemma model with perplexity"""
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
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,  # Enable KV cache for speed
            )
        latency = time.time() - start_time
        
        # Calculate perplexity
        perplexity = self._calculate_perplexity(outputs, inputs)
        
        response = self.gemma_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response = response.split("<start_of_turn>model\n")[-1].strip()
        
        # Clear GPU memory after each generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response, latency, perplexity
    
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
                max_tokens=256
            )
            latency = time.time() - start_time
            return response.choices[0].message.content, latency
        except Exception as e:
            print(f"Error getting ChatGPT response: {e}")
            return f"Error: {str(e)}", 0.0
    
    def _calculate_perplexity(self, outputs, inputs):
        """Calculate perplexity from model outputs"""
        try:
            if hasattr(outputs, 'scores') and outputs.scores:
                # Convert scores to probabilities
                scores = torch.stack(outputs.scores, dim=1)
                log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
                
                # Get log probabilities of generated tokens
                generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]
                token_log_probs = []
                
                for i, token_id in enumerate(generated_tokens):
                    if i < log_probs.shape[1]:
                        token_log_prob = log_probs[0, i, token_id].item()
                        token_log_probs.append(token_log_prob)
                
                if token_log_probs:
                    avg_log_prob = np.mean(token_log_probs)
                    perplexity = np.exp(-avg_log_prob)
                    return float(perplexity)
            return None
        except Exception as e:
            print(f"Warning: Could not calculate perplexity: {e}")
            return None
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using sentence transformers"""
        embeddings = self.semantic_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def calculate_bleu_score(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate BLEU scores (1-gram to 4-gram)"""
        reference_tokens = reference.lower().split()
        generated_tokens = generated.lower().split()
        
        smoothing = SmoothingFunction().method1
        
        bleu_scores = {}
        for n in range(1, 5):
            weights = tuple([1.0/n] * n + [0.0] * (4-n))
            try:
                score = sentence_bleu([reference_tokens], generated_tokens, 
                                     weights=weights, smoothing_function=smoothing)
                bleu_scores[f'bleu_{n}'] = score
            except:
                bleu_scores[f'bleu_{n}'] = 0.0
        
        return bleu_scores
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scores = self.rouge_scorer.score(reference, generated)
        
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure,
        }
    
    def calculate_bert_score(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate BERTScore"""
        try:
            # Suppress warnings during BERTScore calculation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                P, R, F1 = bert_score([generated], [reference], lang='en', verbose=False)
            return {
                'bertscore_precision': P.item(),
                'bertscore_recall': R.item(),
                'bertscore_f1': F1.item(),
            }
        except Exception as e:
            print(f"Warning: BERTScore calculation failed: {e}")
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0,
            }
    
    def calculate_diversity_metrics(self, text: str) -> Dict[str, float]:
        """Calculate text diversity metrics"""
        tokens = text.lower().split()
        
        if len(tokens) == 0:
            return {
                'distinct_1': 0.0,
                'distinct_2': 0.0,
                'lexical_diversity': 0.0
            }
        
        # Distinct-1: unique unigrams / total unigrams
        distinct_1 = len(set(tokens)) / len(tokens)
        
        # Distinct-2: unique bigrams / total bigrams
        bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        
        # Lexical diversity (Type-Token Ratio)
        lexical_diversity = len(set(tokens)) / len(tokens)
        
        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'lexical_diversity': lexical_diversity
        }
    
    def calculate_coherence_score(self, text: str) -> float:
        """Calculate coherence score using sentence similarity"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 1.0  # Single sentence is perfectly coherent
        
        # Encode sentences
        embeddings = self.semantic_model.encode(sentences)
        
        # Calculate average cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i+1]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
            )
            similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 1.0
    
    def calculate_toxicity_score(self, text: str) -> Dict[str, float]:
        """Calculate toxicity scores"""
        if not self.toxicity_model:
            return {
                'toxicity': 0.0,
                'severe_toxicity': 0.0,
                'obscene': 0.0,
                'threat': 0.0,
                'insult': 0.0,
            }
        
        try:
            results = self.toxicity_model.predict(text)
            return {
                'toxicity': float(results['toxicity']),
                'severe_toxicity': float(results['severe_toxicity']),
                'obscene': float(results['obscene']),
                'threat': float(results['threat']),
                'insult': float(results['insult']),
            }
        except Exception as e:
            print(f"Warning: Toxicity calculation failed: {e}")
            return {
                'toxicity': 0.0,
                'severe_toxicity': 0.0,
                'obscene': 0.0,
                'threat': 0.0,
                'insult': 0.0,
            }
    
    def assess_correctness(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Proper correctness assessment using multiple criteria
        """
        # Calculate semantic similarity
        semantic_sim = self.calculate_semantic_similarity(generated, reference)
        
        # Calculate BERT F1 score
        bert_scores = self.calculate_bert_score(generated, reference)
        bert_f1 = bert_scores['bertscore_f1']
        
        # Calculate keyword overlap for domain-specific correctness
        procurement_keywords = [
            'RA 9184', 'procurement', 'bidding', 'contract', 'BAC',
            'evaluation', 'technical', 'financial', 'eligibility',
            'award', 'notice', 'specifications', 'compliance', 'bidder',
            'public', 'government', 'competitive', 'sealed', 'envelope'
        ]
        
        gen_lower = generated.lower()
        ref_lower = reference.lower()
        
        gen_keywords = set(kw.lower() for kw in procurement_keywords if kw.lower() in gen_lower)
        ref_keywords = set(kw.lower() for kw in procurement_keywords if kw.lower() in ref_lower)
        
        if len(ref_keywords) > 0:
            keyword_intersection = gen_keywords.intersection(ref_keywords)
            keyword_precision = len(keyword_intersection) / len(gen_keywords) if len(gen_keywords) > 0 else 0
            keyword_recall = len(keyword_intersection) / len(ref_keywords)
            keyword_f1 = 2 * (keyword_precision * keyword_recall) / (keyword_precision + keyword_recall) if (keyword_precision + keyword_recall) > 0 else 0
        else:
            keyword_precision = keyword_recall = keyword_f1 = 0
        
        # Multi-criteria correctness assessment with realistic thresholds
        correctness_score = (
            semantic_sim * 0.4 +           # 40% semantic similarity
            bert_f1 * 0.4 +                # 40% BERT F1
            keyword_f1 * 0.2               # 20% keyword overlap
        )
        
        # More realistic threshold for correctness (was too low before)
        is_correct = correctness_score > 0.65  # Raised from previous threshold
        
        return {
            'is_correct': is_correct,
            'correctness_score': correctness_score,
            'semantic_similarity': semantic_sim,
            'keyword_precision': keyword_precision,
            'keyword_recall': keyword_recall,
            'keyword_f1': keyword_f1
        }
    
    def calculate_comprehensive_metrics(self, generated: str, reference: str, 
                                       latency: float, perplexity: float = None) -> Dict:
        """Calculate all metrics for a single response"""
        metrics = {}
        
        # Basic info
        metrics['latency'] = latency
        if perplexity:
            metrics['perplexity'] = perplexity
        
        # Correctness assessment (FIXED)
        correctness_assessment = self.assess_correctness(generated, reference)
        metrics.update(correctness_assessment)
        
        # BLEU scores
        bleu_scores = self.calculate_bleu_score(generated, reference)
        metrics.update(bleu_scores)
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge_scores(generated, reference)
        metrics.update(rouge_scores)
        
        # BERTScore
        bert_scores = self.calculate_bert_score(generated, reference)
        metrics.update(bert_scores)
        
        # Diversity metrics
        diversity_metrics = self.calculate_diversity_metrics(generated)
        metrics.update(diversity_metrics)
        
        # Coherence
        metrics['coherence'] = self.calculate_coherence_score(generated)
        
        # Toxicity
        toxicity_scores = self.calculate_toxicity_score(generated)
        metrics.update(toxicity_scores)
        
        return metrics
    
    def run_comparison(self):
        """Run comprehensive comparison with all metrics"""
        print(f"\n{'='*80}")
        print(f"Model Comparison")
        print(f"{'='*80}")
        print(f"Fine-tuned Model: {self.gemma_path}")
        print(f"ChatGPT Model: {self.gpt_model}")
        print(f"Evaluation Dataset: {len(self.eval_data)} questions")
        print(f"{'='*80}\n")
        
        for idx, qa_pair in enumerate(self.eval_data):
            question = qa_pair['question']
            reference = qa_pair['answer']
            
            print(f"\nProcessing [{idx+1}/{len(self.eval_data)}]: {question[:70]}...")
            
            # Get Gemma response
            gemma_response, gemma_latency, gemma_perplexity = self.get_gemma_response(question)
            time.sleep(0.5)
            
            # Get ChatGPT response
            chatgpt_response, chatgpt_latency = self.get_chatgpt_response(question)
            time.sleep(0.5)
            
            # Calculate metrics for both models
            gemma_metrics = self.calculate_comprehensive_metrics(
                gemma_response, reference, gemma_latency, gemma_perplexity
            )
            
            chatgpt_metrics = self.calculate_comprehensive_metrics(
                chatgpt_response, reference, chatgpt_latency, None
            )
            
            # Store results
            result = {
                'question_id': idx + 1,
                'question': question,
                'reference_answer': reference,
                'gemma_response': gemma_response,
                'chatgpt_response': chatgpt_response,
            }
            
            # Add Gemma metrics with prefix
            for key, value in gemma_metrics.items():
                result[f'gemma_{key}'] = value
            
            # Add ChatGPT metrics with prefix
            for key, value in chatgpt_metrics.items():
                result[f'chatgpt_{key}'] = value
            
            self.results.append(result)
            
            # Print summary with semantic similarity
            print(f"  Gemma    - Semantic: {gemma_metrics['semantic_similarity']:.3f} | "
                  f"BERT F1: {gemma_metrics['bertscore_f1']:.3f} | "
                  f"BLEU-4: {gemma_metrics['bleu_4']:.3f} | "
                  f"Correct: {'✓' if gemma_metrics['is_correct'] else '✗'} | "
                  f"Latency: {gemma_latency:.2f}s")
            print(f"  ChatGPT  - Semantic: {chatgpt_metrics['semantic_similarity']:.3f} | "
                  f"BERT F1: {chatgpt_metrics['bertscore_f1']:.3f} | "
                  f"BLEU-4: {chatgpt_metrics['bleu_4']:.3f} | "
                  f"Correct: {'✓' if chatgpt_metrics['is_correct'] else '✗'} | "
                  f"Latency: {chatgpt_latency:.2f}s")
        
        print(f"\n{'='*80}")
        print("Comparison Complete!")
        print(f"{'='*80}\n")
    
    def calculate_aggregate_metrics(self, df: pd.DataFrame, model_prefix: str) -> Dict:
        """Calculate aggregate metrics with FIXED precision, recall, F1, accuracy"""
        metrics = {}
        
        # FIXED: Proper accuracy calculation
        correct_predictions = df[f'{model_prefix}_is_correct'].values
        metrics['accuracy'] = float(np.mean(correct_predictions))
        
        # FIXED: Proper precision, recall, F1 based on keyword metrics
        keyword_precisions = df[f'{model_prefix}_keyword_precision'].values
        keyword_recalls = df[f'{model_prefix}_keyword_recall'].values
        keyword_f1s = df[f'{model_prefix}_keyword_f1'].values
        
        metrics['precision'] = float(np.mean(keyword_precisions))
        metrics['recall'] = float(np.mean(keyword_recalls))
        metrics['f1_score'] = float(np.mean(keyword_f1s))
        
        # Average correctness score
        metrics['avg_correctness_score'] = float(df[f'{model_prefix}_correctness_score'].mean())
        
        # Perplexity (only for Gemma)
        if f'{model_prefix}_perplexity' in df.columns:
            perplexity_values = df[f'{model_prefix}_perplexity'].dropna()
            if len(perplexity_values) > 0:
                metrics['avg_perplexity'] = float(perplexity_values.mean())
                metrics['median_perplexity'] = float(perplexity_values.median())
        
        # Latency
        metrics['avg_latency'] = float(df[f'{model_prefix}_latency'].mean())
        metrics['median_latency'] = float(df[f'{model_prefix}_latency'].median())
        
        # Semantic similarity
        metrics['avg_semantic_similarity'] = float(df[f'{model_prefix}_semantic_similarity'].mean())
        
        # Toxicity
        metrics['avg_toxicity'] = float(df[f'{model_prefix}_toxicity'].mean())
        metrics['max_toxicity'] = float(df[f'{model_prefix}_toxicity'].max())
        
        # BLEU scores
        for n in range(1, 5):
            metrics[f'avg_bleu_{n}'] = float(df[f'{model_prefix}_bleu_{n}'].mean())
        
        # ROUGE scores
        metrics['avg_rouge1_f'] = float(df[f'{model_prefix}_rouge1_f'].mean())
        metrics['avg_rouge2_f'] = float(df[f'{model_prefix}_rouge2_f'].mean())
        metrics['avg_rougeL_f'] = float(df[f'{model_prefix}_rougeL_f'].mean())
        
        # BERTScore
        metrics['avg_bertscore_f1'] = float(df[f'{model_prefix}_bertscore_f1'].mean())
        metrics['avg_bertscore_precision'] = float(df[f'{model_prefix}_bertscore_precision'].mean())
        metrics['avg_bertscore_recall'] = float(df[f'{model_prefix}_bertscore_recall'].mean())
        
        # Diversity and Coherence
        metrics['avg_distinct_1'] = float(df[f'{model_prefix}_distinct_1'].mean())
        metrics['avg_distinct_2'] = float(df[f'{model_prefix}_distinct_2'].mean())
        metrics['avg_lexical_diversity'] = float(df[f'{model_prefix}_lexical_diversity'].mean())
        metrics['avg_coherence'] = float(df[f'{model_prefix}_coherence'].mean())
        
        return metrics
    
    def generate_report(self, output_dir: str = "comparison_result_v2"):
        """Generate comprehensive comparison report with all metrics"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate aggregate statistics
        gemma_stats = self.calculate_aggregate_metrics(df, 'gemma')
        chatgpt_stats = self.calculate_aggregate_metrics(df, 'chatgpt')
        
        stats = {
            'gemma': gemma_stats,
            'chatgpt': chatgpt_stats
        }
        
        # Generate visualizations
        self._create_advanced_visualizations(df, output_dir, timestamp)
        
        # Save detailed results
        df.to_csv(f"{output_dir}/detailed_results_{timestamp}.csv", index=False)
        
        # Save statistics
        with open(f"{output_dir}/statistics_{timestamp}.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate text report
        report = self._generate_advanced_report(stats, df)
        with open(f"{output_dir}/comprehensive_report_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_dir}/")
        print(f"{'='*80}\n")
        print(report)
        
        return stats
    
    def _create_advanced_visualizations(self, df: pd.DataFrame, output_dir: str, timestamp: str):
        """Create comprehensive visualization suite"""
        sns.set_style("whitegrid")
        
        # Create multiple visualization figures
        
        # Figure 1: Core Performance Metrics
        fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig1.suptitle('Core Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy
        ax = axes[0, 0]
        accuracy_data = [
            df['gemma_is_correct'].mean(),
            df['chatgpt_is_correct'].mean()
        ]
        bars = ax.bar(['Fine-tuned Gemma', 'ChatGPT'], accuracy_data, 
                     color=['#3498db', '#e74c3c'], alpha=0.7)
        ax.set_ylabel('Accuracy Score')
        ax.set_title('Accuracy (Correctness Rate)')
        ax.set_ylim([0, 1])
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        # Semantic Similarity
        ax = axes[0, 1]
        ax.boxplot([df['gemma_semantic_similarity'], df['chatgpt_semantic_similarity']],
                   labels=['Fine-tuned Gemma', 'ChatGPT'])
        ax.set_ylabel('Semantic Similarity')
        ax.set_title('Semantic Similarity Distribution')
        ax.grid(True, alpha=0.3)
        
        # BERTScore F1
        ax = axes[0, 2]
        ax.boxplot([df['gemma_bertscore_f1'], df['chatgpt_bertscore_f1']],
                   labels=['Fine-tuned Gemma', 'ChatGPT'])
        ax.set_ylabel('BERTScore F1')
        ax.set_title('BERTScore F1 Distribution')
        ax.grid(True, alpha=0.3)
        
        # BLEU-4 Scores
        ax = axes[1, 0]
        ax.boxplot([df['gemma_bleu_4'], df['chatgpt_bleu_4']],
                   labels=['Fine-tuned Gemma', 'ChatGPT'])
        ax.set_ylabel('BLEU-4 Score')
        ax.set_title('BLEU-4 Score Distribution')
        ax.grid(True, alpha=0.3)
        
        # Latency
        ax = axes[1, 1]
        ax.boxplot([df['gemma_latency'], df['chatgpt_latency']],
                   labels=['Fine-tuned Gemma', 'ChatGPT'])
        ax.set_ylabel('Latency (seconds)')
        ax.set_title('Response Latency Distribution')
        ax.grid(True, alpha=0.3)
        
        # Precision, Recall, F1
        ax = axes[1, 2]
        x = np.arange(3)
        width = 0.35
        
        gemma_prf = [df['gemma_keyword_precision'].mean(), 
                     df['gemma_keyword_recall'].mean(), 
                     df['gemma_keyword_f1'].mean()]
        chatgpt_prf = [df['chatgpt_keyword_precision'].mean(), 
                       df['chatgpt_keyword_recall'].mean(), 
                       df['chatgpt_keyword_f1'].mean()]
        
        ax.bar(x - width/2, gemma_prf, width, label='Fine-tuned Gemma', alpha=0.7, color='#3498db')
        ax.bar(x + width/2, chatgpt_prf, width, label='ChatGPT', alpha=0.7, color='#e74c3c')
        
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall, F1-Score')
        ax.set_xticks(x)
        ax.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/core_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Diversity and Coherence
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig2.suptitle('Diversity and Coherence Metrics', fontsize=16, fontweight='bold')
        
        # Distinct-1
        ax = axes[0, 0]
        ax.boxplot([df['gemma_distinct_1'], df['chatgpt_distinct_1']],
                   labels=['Fine-tuned Gemma', 'ChatGPT'])
        ax.set_ylabel('Distinct-1 Score')
        ax.set_title('Unigram Diversity')
        ax.grid(True, alpha=0.3)
        
        # Distinct-2
        ax = axes[0, 1]
        ax.boxplot([df['gemma_distinct_2'], df['chatgpt_distinct_2']],
                   labels=['Fine-tuned Gemma', 'ChatGPT'])
        ax.set_ylabel('Distinct-2 Score')
        ax.set_title('Bigram Diversity')
        ax.grid(True, alpha=0.3)
        
        # Lexical Diversity
        ax = axes[1, 0]
        ax.boxplot([df['gemma_lexical_diversity'], df['chatgpt_lexical_diversity']],
                   labels=['Fine-tuned Gemma', 'ChatGPT'])
        ax.set_ylabel('Lexical Diversity')
        ax.set_title('Type-Token Ratio')
        ax.grid(True, alpha=0.3)
        
        # Coherence
        ax = axes[1, 1]
        ax.boxplot([df['gemma_coherence'], df['chatgpt_coherence']],
                   labels=['Fine-tuned Gemma', 'ChatGPT'])
        ax.set_ylabel('Coherence Score')
        ax.set_title('Text Coherence')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/diversity_coherence_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Detailed Metric Comparison Radar Chart
        fig3, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        categories = ['Accuracy', 'Semantic Sim', 'BLEU-4', 'ROUGE-L', 'BERTScore', 
                     'Diversity', 'Coherence', 'Speed']
        
        gemma_values = [
            df['gemma_is_correct'].mean(),
            df['gemma_semantic_similarity'].mean(),
            df['gemma_bleu_4'].mean(),
            df['gemma_rougeL_f'].mean(),
            df['gemma_bertscore_f1'].mean(),
            df['gemma_distinct_1'].mean(),
            df['gemma_coherence'].mean(),
            1 - (df['gemma_latency'].mean() / (df['gemma_latency'].mean() + df['chatgpt_latency'].mean()))
        ]
        
        chatgpt_values = [
            df['chatgpt_is_correct'].mean(),
            df['chatgpt_semantic_similarity'].mean(),
            df['chatgpt_bleu_4'].mean(),
            df['chatgpt_rougeL_f'].mean(),
            df['chatgpt_bertscore_f1'].mean(),
            df['chatgpt_distinct_1'].mean(),
            df['chatgpt_coherence'].mean(),
            1 - (df['chatgpt_latency'].mean() / (df['gemma_latency'].mean() + df['chatgpt_latency'].mean()))
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        gemma_values += gemma_values[:1]
        chatgpt_values += chatgpt_values[:1]
        angles += angles[:1]
        
        ax.plot(angles, gemma_values, 'o-', linewidth=2, label='Fine-tuned Gemma', color='#3498db')
        ax.fill(angles, gemma_values, alpha=0.25, color='#3498db')
        ax.plot(angles, chatgpt_values, 'o-', linewidth=2, label='ChatGPT', color='#e74c3c')
        ax.fill(angles, chatgpt_values, alpha=0.25, color='#e74c3c')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Comprehensive Metric Comparison (Radar Chart)', 
                    size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/radar_chart_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_advanced_report(self, stats: Dict, df: pd.DataFrame) -> str:
        """Generate comprehensive text report"""
        report = []
        report.append("=" * 90)
        report.append("COMPREHENSIVE MODEL COMPARISON REPORT V2")
        report.append("Advanced Metrics for Philippine Procurement Domain Knowledge")
        report.append("=" * 90)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nModels Compared:")
        report.append(f"  1. Fine-tuned Gemma 3 1B (from {self.gemma_path})")
        report.append(f"  2. ChatGPT GPT-4o-mini (OpenAI)")
        report.append(f"\nEvaluation Dataset: {len(self.eval_data)} Q&A pairs")
        
        report.append("\n" + "=" * 90)
        report.append("1. GENERAL PERFORMANCE METRICS (CORRECTED)")
        report.append("=" * 90)
        
        # Accuracy
        report.append("\n📊 ACCURACY (Percentage of Correct Answers)")
        report.append("-" * 50)
        report.append(f"Fine-tuned Gemma: {stats['gemma']['accuracy']:.4f} ({stats['gemma']['accuracy']*100:.2f}%)")
        report.append(f"ChatGPT:          {stats['chatgpt']['accuracy']:.4f} ({stats['chatgpt']['accuracy']*100:.2f}%)")
        diff = stats['gemma']['accuracy'] - stats['chatgpt']['accuracy']
        winner = "Fine-tuned Gemma" if diff > 0 else "ChatGPT"
        report.append(f"→ Winner: {winner} (Δ = {abs(diff):.4f})")
        
        # Precision, Recall, F1 (FIXED)
        report.append("\n📊 PRECISION, RECALL, F1-SCORE (Keyword-based)")
        report.append("-" * 50)
        report.append(f"Fine-tuned Gemma:")
        report.append(f"  Precision: {stats['gemma']['precision']:.4f}")
        report.append(f"  Recall:    {stats['gemma']['recall']:.4f}")
        report.append(f"  F1-Score:  {stats['gemma']['f1_score']:.4f}")
        report.append(f"\nChatGPT:")
        report.append(f"  Precision: {stats['chatgpt']['precision']:.4f}")
        report.append(f"  Recall:    {stats['chatgpt']['recall']:.4f}")
        report.append(f"  F1-Score:  {stats['chatgpt']['f1_score']:.4f}")
        
        f1_diff = stats['gemma']['f1_score'] - stats['chatgpt']['f1_score']
        winner = "Fine-tuned Gemma" if f1_diff > 0 else "ChatGPT"
        report.append(f"→ Winner (F1): {winner} (Δ = {abs(f1_diff):.4f})")
        
        # Semantic Similarity
        report.append("\n🔗 SEMANTIC SIMILARITY (Higher is Better)")
        report.append("-" * 50)
        report.append(f"Fine-tuned Gemma: {stats['gemma']['avg_semantic_similarity']:.4f}")
        report.append(f"ChatGPT:          {stats['chatgpt']['avg_semantic_similarity']:.4f}")
        
        sem_diff = stats['gemma']['avg_semantic_similarity'] - stats['chatgpt']['avg_semantic_similarity']
        winner = "Fine-tuned Gemma" if sem_diff > 0 else "ChatGPT"
        report.append(f"→ Winner: {winner} (Δ = {abs(sem_diff):.4f})")
        
        # Perplexity
        if 'avg_perplexity' in stats['gemma']:
            report.append("\n📊 PERPLEXITY (Language Model Quality - Lower is Better)")
            report.append("-" * 50)
            report.append(f"Fine-tuned Gemma:")
            report.append(f"  Average:  {stats['gemma']['avg_perplexity']:.2f}")
            report.append(f"  Median:   {stats['gemma']['median_perplexity']:.2f}")
            report.append(f"\nNote: Lower perplexity indicates better language modeling.")
            report.append(f"      ChatGPT perplexity unavailable (API limitation).")
        
        # Latency
        report.append("\n⚡ LATENCY (Response Speed - Lower is Better)")
        report.append("-" * 50)
        report.append(f"Fine-tuned Gemma:")
        report.append(f"  Average:  {stats['gemma']['avg_latency']:.3f}s")
        report.append(f"  Median:   {stats['gemma']['median_latency']:.3f}s")
        report.append(f"\nChatGPT:")
        report.append(f"  Average:  {stats['chatgpt']['avg_latency']:.3f}s")
        report.append(f"  Median:   {stats['chatgpt']['median_latency']:.3f}s")
        
        latency_diff = stats['gemma']['avg_latency'] - stats['chatgpt']['avg_latency']
        winner = "Fine-tuned Gemma" if latency_diff < 0 else "ChatGPT"
        speedup = abs(latency_diff)
        report.append(f"→ Winner: {winner} ({speedup:.3f}s faster on average)")
        
        # Continue with other sections...
        report.append("\n" + "=" * 90)
        report.append("2. TEXT-SPECIFIC AND SEMANTIC METRICS")
        report.append("=" * 90)
        
        # BLEU Scores
        report.append("\n📝 BLEU SCORES (N-gram Overlap)")
        report.append("-" * 50)
        report.append(f"Fine-tuned Gemma:")
        for n in range(1, 5):
            report.append(f"  BLEU-{n}: {stats['gemma'][f'avg_bleu_{n}']:.4f}")
        report.append(f"\nChatGPT:")
        for n in range(1, 5):
            report.append(f"  BLEU-{n}: {stats['chatgpt'][f'avg_bleu_{n}']:.4f}")
        
        bleu4_diff = stats['gemma']['avg_bleu_4'] - stats['chatgpt']['avg_bleu_4']
        winner = "Fine-tuned Gemma" if bleu4_diff > 0 else "ChatGPT"
        report.append(f"→ Winner (BLEU-4): {winner} (Δ = {abs(bleu4_diff):.4f})")
        
        # BERTScore
        report.append("\n🤖 BERTSCORE (Semantic Similarity with Embeddings)")
        report.append("-" * 50)
        report.append(f"Fine-tuned Gemma:")
        report.append(f"  Precision: {stats['gemma']['avg_bertscore_precision']:.4f}")
        report.append(f"  Recall:    {stats['gemma']['avg_bertscore_recall']:.4f}")
        report.append(f"  F1:        {stats['gemma']['avg_bertscore_f1']:.4f}")
        report.append(f"\nChatGPT:")
        report.append(f"  Precision: {stats['chatgpt']['avg_bertscore_precision']:.4f}")
        report.append(f"  Recall:    {stats['chatgpt']['avg_bertscore_recall']:.4f}")
        report.append(f"  F1:        {stats['chatgpt']['avg_bertscore_f1']:.4f}")
        
        bert_diff = stats['gemma']['avg_bertscore_f1'] - stats['chatgpt']['avg_bertscore_f1']
        winner = "Fine-tuned Gemma" if bert_diff > 0 else "ChatGPT"
        report.append(f"→ Winner (BERTScore F1): {winner} (Δ = {abs(bert_diff):.4f})")
        
        # Overall Assessment
        report.append("\n" + "=" * 90)
        report.append("3. OVERALL ASSESSMENT")
        report.append("=" * 90)
        
        # Calculate wins
        wins = {'gemma': 0, 'chatgpt': 0}
        
        categories = [
            ('Accuracy', stats['gemma']['accuracy'] > stats['chatgpt']['accuracy']),
            ('F1-Score', stats['gemma']['f1_score'] > stats['chatgpt']['f1_score']),
            ('Semantic Similarity', stats['gemma']['avg_semantic_similarity'] > stats['chatgpt']['avg_semantic_similarity']),
            ('Latency', stats['gemma']['avg_latency'] < stats['chatgpt']['avg_latency']),
            ('BLEU-4', stats['gemma']['avg_bleu_4'] > stats['chatgpt']['avg_bleu_4']),
            ('BERTScore', stats['gemma']['avg_bertscore_f1'] > stats['chatgpt']['avg_bertscore_f1']),
        ]
        
        report.append("\nMetric-by-Metric Winners:")
        report.append("-" * 50)
        for category, gemma_wins in categories:
            winner_name = "✓ Fine-tuned Gemma" if gemma_wins else "✓ ChatGPT"
            report.append(f"  {category:.<25} {winner_name}")
            if gemma_wins:
                wins['gemma'] += 1
            else:
                wins['chatgpt'] += 1
        
        report.append(f"\n{'='*50}")
        report.append(f"FINAL SCORE:")
        report.append(f"  Fine-tuned Gemma: {wins['gemma']}/6 categories")
        report.append(f"  ChatGPT:          {wins['chatgpt']}/6 categories")
        report.append(f"{'='*50}")
        
        # Recommendations
        report.append("\n" + "=" * 90)
        report.append("4. RECOMMENDATIONS")
        report.append("=" * 90)
        
        if wins['gemma'] >= 3:
            report.append("\n✅ DEPLOYMENT RECOMMENDATION: Fine-tuned Gemma")
            report.append("-" * 50)
            report.append("Your fine-tuned model shows competitive/superior performance.")
        else:
            report.append("\n📊 RECOMMENDATION: Continue Development")
            report.append("-" * 50)
            report.append("Consider additional fine-tuning or data augmentation.")
        
        report.append("\n" + "=" * 90)
        report.append("5. EVALUATION IMPROVEMENTS IN V2")
        report.append("=" * 90)
        report.append("\n✅ FIXED ISSUES:")
        report.append("  • Realistic accuracy thresholds (was 100%, now proper evaluation)")
        report.append("  • Proper precision/recall/F1 calculation")
        report.append("  • Added semantic similarity in output display")
        report.append("  • Improved keyword-based domain evaluation")
        report.append("  • Better multi-criteria correctness assessment")
        report.append("  • Fixed torch_dtype deprecation warning")
        report.append("  • Removed invalid early_stopping flag")
        report.append("  • Suppressed unnecessary model warnings")
        
        report.append("\n" + "=" * 90)
        report.append("END OF COMPREHENSIVE REPORT V2")
        report.append("=" * 90)
        report.append(f"\nFor detailed per-question results, see: detailed_results_*.csv")
        report.append(f"For visualizations, see: *.png files in comparison_result_v2/ directory")
        
        return "\n".join(report)


def main():
    """Main execution function"""
    # Configuration
    GEMMA_MODEL_PATH = "gemma3_1b_procurement_weska_v2"
    EVAL_FILE = "eval_qa_pairs.jsonl"
    OPENAI_API_KEY = "REPLACED_FOR_SECURITY"
    
    if not OPENAI_API_KEY:
        print("=" * 80)
        print("ERROR: OpenAI API Key Not Found")
        print("=" * 80)
        print("\nPlease set your OpenAI API key in the script or as environment variable.")
        print("=" * 80)
        return
    
    try:
        # Initialize comparison
        print("\n" + "=" * 80)
        print("Initializing Advanced Model Comparison System V2")
        print("=" * 80)
        
        comparison = AdvancedModelComparison(
            gemma_path=GEMMA_MODEL_PATH,
            openai_api_key=OPENAI_API_KEY,
            eval_file=EVAL_FILE
        )
        
        # Run comparison
        comparison.run_comparison()
        
        # Generate comprehensive report
        stats = comparison.generate_report()
        
        print("\n" + "=" * 80)
        print("✅ Comparison Complete!")
        print("=" * 80)
        print("\nGenerated files in comparison_result_v2/:")
        print("  📊 detailed_results_*.csv - All responses and metrics")
        print("  📈 core_metrics_*.png - Performance visualizations")
        print("  📈 diversity_coherence_*.png - Text quality metrics")
        print("  📈 radar_chart_*.png - Overall comparison")
        print("  📝 comprehensive_report_*.txt - Full analysis")
        print("  📋 statistics_*.json - Raw statistics")
        print("\n🔧 IMPROVEMENTS IN V2:")
        print("  ✅ Fixed unrealistic 100% accuracy issue")
        print("  ✅ Proper precision/recall/F1 calculation")
        print("  ✅ Added semantic similarity to display")
        print("  ✅ Better GPU memory management")
        print("  ✅ Suppressed model warnings")
        print("  ✅ More realistic evaluation thresholds")
        print("=" * 80 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print(f"  1. Model folder exists: {GEMMA_MODEL_PATH}")
        print(f"  2. Evaluation file exists: {EVAL_FILE}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()