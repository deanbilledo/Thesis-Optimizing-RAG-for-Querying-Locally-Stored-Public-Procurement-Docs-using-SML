import json
import time
import os
import logging
import warnings
from pathlib import Path
import numpy as np
from typing import List, Dict, Tuple, Set
import pandas as pd
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Comprehensive warning suppression
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Set logging levels to suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)

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
        Initialize advanced comparison framework with comprehensive metrics V3
        
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
        
        # V3: Use a better semantic similarity model
        print("Loading enhanced semantic similarity model...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use a more domain-specific and powerful model
            self.semantic_model = SentenceTransformer('paraphrase-mpnet-base-v2')
            # Also load a second model for cross-validation
            self.semantic_model_alt = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize toxicity detector
        if DETOXIFY_AVAILABLE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.toxicity_model = Detoxify('original')
        else:
            self.toxicity_model = None
        
        # Load Gemma model with enhanced error handling
        print(f"Loading fine-tuned Gemma model from {gemma_path}...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gemma_tokenizer = AutoTokenizer.from_pretrained(gemma_path)
            self.gemma_model = AutoModelForCausalLM.from_pretrained(
                gemma_path,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        # Ensure model is in eval mode for inference
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
                max_new_tokens=300,  # Slightly increased for better responses
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.gemma_tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,  # Enable KV cache for speed
                repetition_penalty=1.1,  # V3: Add repetition penalty
            )
        latency = time.time() - start_time
        
        # Calculate perplexity
        perplexity = self._calculate_perplexity(outputs, inputs)
        
        response = self.gemma_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response = response.split("<start_of_turn>model\n")[-1].strip()
        
        # V3: Clean up response better
        response = self._clean_response(response)
        
        # Clear GPU memory after each generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response, latency, perplexity
    
    def _clean_response(self, response: str) -> str:
        """V3: Better response cleaning"""
        # Remove any remaining template artifacts
        response = re.sub(r'<[^>]*>', '', response)
        # Remove excessive whitespace
        response = re.sub(r'\s+', ' ', response).strip()
        # Remove common repetitive patterns
        lines = response.split('\n')
        cleaned_lines = []
        prev_line = ""
        for line in lines:
            line = line.strip()
            if line and line != prev_line:  # Remove duplicate consecutive lines
                cleaned_lines.append(line)
                prev_line = line
        
        return '\n'.join(cleaned_lines) if cleaned_lines else response
    
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
                max_tokens=300  # Matched with Gemma
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
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """V3: Enhanced semantic similarity with multiple models"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Primary model
            embeddings1 = self.semantic_model.encode([text1, text2])
            similarity1 = cosine_similarity([embeddings1[0]], [embeddings1[1]])[0][0]
            
            # Secondary model for validation
            embeddings2 = self.semantic_model_alt.encode([text1, text2])
            similarity2 = cosine_similarity([embeddings2[0]], [embeddings2[1]])[0][0]
            
            # Average the two for more robust measure
            avg_similarity = (similarity1 + similarity2) / 2
            
            return {
                'semantic_similarity': float(avg_similarity),
                'semantic_similarity_primary': float(similarity1),
                'semantic_similarity_secondary': float(similarity2),
                'semantic_confidence': 1.0 - abs(similarity1 - similarity2)  # Confidence based on agreement
            }
    
    def calculate_bleu_score(self, generated: str, reference: str) -> Dict[str, float]:
        """V3: Enhanced BLEU calculation with better preprocessing"""
        # Better tokenization
        reference_tokens = reference.lower().replace('.', ' .').replace(',', ' ,').split()
        generated_tokens = generated.lower().replace('.', ' .').replace(',', ' ,').split()
        
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
        """Calculate BERTScore with warning suppression"""
        try:
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
        """V3: Enhanced diversity metrics"""
        tokens = text.lower().split()
        
        if len(tokens) == 0:
            return {
                'distinct_1': 0.0,
                'distinct_2': 0.0,
                'distinct_3': 0.0,  # V3: Added trigrams
                'lexical_diversity': 0.0,
                'avg_word_length': 0.0  # V3: Added average word length
            }
        
        # Distinct-1: unique unigrams / total unigrams
        distinct_1 = len(set(tokens)) / len(tokens)
        
        # Distinct-2: unique bigrams / total bigrams
        bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        
        # V3: Distinct-3: unique trigrams / total trigrams
        trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
        distinct_3 = len(set(trigrams)) / len(trigrams) if trigrams else 0.0
        
        # Lexical diversity (Type-Token Ratio)
        lexical_diversity = len(set(tokens)) / len(tokens)
        
        # V3: Average word length
        avg_word_length = np.mean([len(word) for word in tokens])
        
        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'distinct_3': distinct_3,
            'lexical_diversity': lexical_diversity,
            'avg_word_length': avg_word_length
        }
    
    def calculate_coherence_score(self, text: str) -> float:
        """V3: Enhanced coherence calculation"""
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 10]
        
        if len(sentences) < 2:
            return 1.0  # Single sentence is perfectly coherent
        
        # Use the primary semantic model for coherence
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        """Calculate toxicity scores with warning suppression"""
        if not self.toxicity_model:
            return {
                'toxicity': 0.0,
                'severe_toxicity': 0.0,
                'obscene': 0.0,
                'threat': 0.0,
                'insult': 0.0,
            }
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
        """V3: Enhanced correctness assessment with improved criteria"""
        # Calculate enhanced semantic similarity
        semantic_metrics = self.calculate_semantic_similarity(generated, reference)
        semantic_sim = semantic_metrics['semantic_similarity']
        confidence = semantic_metrics['semantic_confidence']
        
        # Calculate BERT F1 score
        bert_scores = self.calculate_bert_score(generated, reference)
        bert_f1 = bert_scores['bertscore_f1']
        
        # V3: Enhanced keyword analysis with weighted importance
        high_importance_keywords = [
            'RA 9184', 'Republic Act 9184', 'procurement law', 'government procurement',
            'public bidding', 'BAC', 'Bids and Awards Committee'
        ]
        
        medium_importance_keywords = [
            'procurement', 'bidding', 'contract', 'evaluation', 'technical', 'financial',
            'eligibility', 'award', 'notice', 'specifications', 'compliance', 'bidder',
            'public', 'government', 'competitive', 'sealed', 'envelope'
        ]
        
        gen_lower = generated.lower()
        ref_lower = reference.lower()
        
        # High importance keyword matching
        gen_high = sum(1 for kw in high_importance_keywords if kw.lower() in gen_lower)
        ref_high = sum(1 for kw in high_importance_keywords if kw.lower() in ref_lower)
        
        # Medium importance keyword matching
        gen_medium = sum(1 for kw in medium_importance_keywords if kw.lower() in gen_lower)
        ref_medium = sum(1 for kw in medium_importance_keywords if kw.lower() in ref_lower)
        
        # Calculate weighted keyword scores
        if ref_high + ref_medium > 0:
            high_weight = 0.7
            medium_weight = 0.3
            
            high_recall = gen_high / ref_high if ref_high > 0 else 0
            medium_recall = gen_medium / ref_medium if ref_medium > 0 else 0
            
            total_gen = gen_high + gen_medium
            high_precision = gen_high / total_gen if total_gen > 0 else 0
            medium_precision = gen_medium / total_gen if total_gen > 0 else 0
            
            keyword_recall = (high_recall * high_weight + medium_recall * medium_weight)
            keyword_precision = (high_precision * high_weight + medium_precision * medium_weight)
            keyword_f1 = 2 * (keyword_precision * keyword_recall) / (keyword_precision + keyword_recall) if (keyword_precision + keyword_recall) > 0 else 0
        else:
            keyword_precision = keyword_recall = keyword_f1 = 0
        
        # V3: Enhanced multi-criteria correctness assessment
        correctness_score = (
            semantic_sim * 0.35 +          # 35% semantic similarity
            bert_f1 * 0.35 +               # 35% BERT F1
            keyword_f1 * 0.25 +            # 25% keyword overlap
            confidence * 0.05              # 5% confidence bonus
        )
        
        # V3: Adaptive threshold based on question complexity
        question_length = len(reference.split())
        if question_length < 20:
            threshold = 0.70  # Higher threshold for simple questions
        elif question_length < 50:
            threshold = 0.65  # Medium threshold
        else:
            threshold = 0.60  # Lower threshold for complex questions
        
        is_correct = correctness_score > threshold
        
        return {
            'is_correct': is_correct,
            'correctness_score': correctness_score,
            'semantic_similarity': semantic_sim,
            'semantic_confidence': confidence,
            'keyword_precision': keyword_precision,
            'keyword_recall': keyword_recall,
            'keyword_f1': keyword_f1,
            'adaptive_threshold': threshold
        }
    
    def calculate_comprehensive_metrics(self, generated: str, reference: str, 
                                       latency: float, perplexity: float = None) -> Dict:
        """V3: Enhanced comprehensive metrics calculation"""
        metrics = {}
        
        # Basic info
        metrics['latency'] = latency
        if perplexity:
            metrics['perplexity'] = perplexity
        
        # V3: Enhanced correctness assessment
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
        
        # V3: Enhanced diversity metrics
        diversity_metrics = self.calculate_diversity_metrics(generated)
        metrics.update(diversity_metrics)
        
        # Coherence
        metrics['coherence'] = self.calculate_coherence_score(generated)
        
        # Toxicity
        toxicity_scores = self.calculate_toxicity_score(generated)
        metrics.update(toxicity_scores)
        
        # V3: Response quality metrics
        metrics['response_length'] = len(generated.split())
        metrics['reference_length'] = len(reference.split())
        metrics['length_ratio'] = metrics['response_length'] / max(metrics['reference_length'], 1)
        
        return metrics
    
    def run_comparison(self):
        """Run comprehensive comparison with all metrics"""
        print(f"\n{'='*80}")
        print(f"Advanced Model Comparison V3 with Enhanced Evaluation")
        print(f"{'='*80}")
        print(f"Fine-tuned Model: {self.gemma_path}")
        print(f"ChatGPT Model: {self.gpt_model}")
        print(f"Evaluation Dataset: {len(self.eval_data)} questions")
        print(f"Semantic Models: paraphrase-mpnet-base-v2 + all-mpnet-base-v2")
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
            
            # V3: Enhanced output display
            print(f"  Gemma    - Semantic: {gemma_metrics['semantic_similarity']:.3f} "
                  f"(±{1-gemma_metrics['semantic_confidence']:.3f}) | "
                  f"BERT F1: {gemma_metrics['bertscore_f1']:.3f} | "
                  f"BLEU-4: {gemma_metrics['bleu_4']:.3f} | "
                  f"Correct: {'✓' if gemma_metrics['is_correct'] else '✗'} | "
                  f"Latency: {gemma_latency:.2f}s")
            print(f"  ChatGPT  - Semantic: {chatgpt_metrics['semantic_similarity']:.3f} "
                  f"(±{1-chatgpt_metrics['semantic_confidence']:.3f}) | "
                  f"BERT F1: {chatgpt_metrics['bertscore_f1']:.3f} | "
                  f"BLEU-4: {chatgpt_metrics['bleu_4']:.3f} | "
                  f"Correct: {'✓' if chatgpt_metrics['is_correct'] else '✗'} | "
                  f"Latency: {chatgpt_latency:.2f}s")
        
        print(f"\n{'='*80}")
        print("V3 Comparison Complete!")
        print(f"{'='*80}\n")
    
    def calculate_aggregate_metrics(self, df: pd.DataFrame, model_prefix: str) -> Dict:
        """V3: Enhanced aggregate metrics calculation"""
        metrics = {}
        
        # Corrected accuracy calculation
        correct_predictions = df[f'{model_prefix}_is_correct'].values
        metrics['accuracy'] = float(np.mean(correct_predictions))
        
        # Enhanced precision, recall, F1 based on keyword metrics
        keyword_precisions = df[f'{model_prefix}_keyword_precision'].values
        keyword_recalls = df[f'{model_prefix}_keyword_recall'].values
        keyword_f1s = df[f'{model_prefix}_keyword_f1'].values
        
        metrics['precision'] = float(np.mean(keyword_precisions))
        metrics['recall'] = float(np.mean(keyword_recalls))
        metrics['f1_score'] = float(np.mean(keyword_f1s))
        
        # V3: Enhanced semantic metrics
        metrics['avg_correctness_score'] = float(df[f'{model_prefix}_correctness_score'].mean())
        metrics['avg_semantic_similarity'] = float(df[f'{model_prefix}_semantic_similarity'].mean())
        metrics['avg_semantic_confidence'] = float(df[f'{model_prefix}_semantic_confidence'].mean())
        
        # Perplexity (only for Gemma)
        if f'{model_prefix}_perplexity' in df.columns:
            perplexity_values = df[f'{model_prefix}_perplexity'].dropna()
            if len(perplexity_values) > 0:
                metrics['avg_perplexity'] = float(perplexity_values.mean())
                metrics['median_perplexity'] = float(perplexity_values.median())
        
        # Latency
        metrics['avg_latency'] = float(df[f'{model_prefix}_latency'].mean())
        metrics['median_latency'] = float(df[f'{model_prefix}_latency'].median())
        
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
        
        # V3: Enhanced diversity and coherence
        metrics['avg_distinct_1'] = float(df[f'{model_prefix}_distinct_1'].mean())
        metrics['avg_distinct_2'] = float(df[f'{model_prefix}_distinct_2'].mean())
        metrics['avg_distinct_3'] = float(df[f'{model_prefix}_distinct_3'].mean())
        metrics['avg_lexical_diversity'] = float(df[f'{model_prefix}_lexical_diversity'].mean())
        metrics['avg_coherence'] = float(df[f'{model_prefix}_coherence'].mean())
        metrics['avg_word_length'] = float(df[f'{model_prefix}_avg_word_length'].mean())
        
        # V3: Response quality
        metrics['avg_response_length'] = float(df[f'{model_prefix}_response_length'].mean())
        metrics['avg_length_ratio'] = float(df[f'{model_prefix}_length_ratio'].mean())
        
        return metrics
    
    def generate_report(self, output_dir: str = "comparison_result_v3"):
        """Generate comprehensive comparison report with all V3 enhancements"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate aggregate statistics
        gemma_stats = self.calculate_aggregate_metrics(df, 'gemma')
        chatgpt_stats = self.calculate_aggregate_metrics(df, 'chatgpt')
        
        stats = {
            'gemma': gemma_stats,
            'chatgpt': chatgpt_stats,
            'evaluation_version': 'V3',
            'semantic_models': ['paraphrase-mpnet-base-v2', 'all-mpnet-base-v2'],
            'enhancements': [
                'Dual semantic similarity models',
                'Adaptive correctness thresholds',
                'Enhanced keyword weighting',
                'Improved response cleaning',
                'Trigram diversity metrics',
                'Semantic confidence scoring'
            ]
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
        """V3: Enhanced visualization suite"""
        sns.set_style("whitegrid")
        plt.style.use('default')
        
        # Figure 1: V3 Core Performance Metrics
        fig1, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig1.suptitle('V3 Enhanced Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        # Accuracy with confidence intervals
        ax = axes[0, 0]
        accuracy_data = [df['gemma_is_correct'].mean(), df['chatgpt_is_correct'].mean()]
        accuracy_std = [df['gemma_is_correct'].std(), df['chatgpt_is_correct'].std()]
        bars = ax.bar(['Fine-tuned Gemma', 'ChatGPT'], accuracy_data, 
                     yerr=accuracy_std, capsize=5, color=['#3498db', '#e74c3c'], alpha=0.7)
        ax.set_ylabel('Accuracy Score')
        ax.set_title('Accuracy (with std dev)')
        ax.set_ylim([0, 1])
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + accuracy_std[i] + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        # Semantic Similarity with Confidence
        ax = axes[0, 1]
        ax.scatter(df['gemma_semantic_similarity'], df['gemma_semantic_confidence'], 
                  alpha=0.6, label='Fine-tuned Gemma', s=30, color='#3498db')
        ax.scatter(df['chatgpt_semantic_similarity'], df['chatgpt_semantic_confidence'], 
                  alpha=0.6, label='ChatGPT', s=30, color='#e74c3c')
        ax.set_xlabel('Semantic Similarity')
        ax.set_ylabel('Semantic Confidence')
        ax.set_title('Semantic Similarity vs Confidence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Enhanced Diversity Metrics
        ax = axes[0, 2]
        diversity_metrics = ['distinct_1', 'distinct_2', 'distinct_3']
        x = np.arange(len(diversity_metrics))
        width = 0.35
        
        gemma_div = [df[f'gemma_{metric}'].mean() for metric in diversity_metrics]
        chatgpt_div = [df[f'chatgpt_{metric}'].mean() for metric in diversity_metrics]
        
        ax.bar(x - width/2, gemma_div, width, label='Fine-tuned Gemma', alpha=0.7, color='#3498db')
        ax.bar(x + width/2, chatgpt_div, width, label='ChatGPT', alpha=0.7, color='#e74c3c')
        
        ax.set_ylabel('Diversity Score')
        ax.set_title('N-gram Diversity (V3)')
        ax.set_xticks(x)
        ax.set_xticklabels(['Unigrams', 'Bigrams', 'Trigrams'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Response Length Distribution
        ax = axes[1, 0]
        ax.hist(df['gemma_response_length'], bins=20, alpha=0.6, label='Fine-tuned Gemma', color='#3498db')
        ax.hist(df['chatgpt_response_length'], bins=20, alpha=0.6, label='ChatGPT', color='#e74c3c')
        ax.set_xlabel('Response Length (words)')
        ax.set_ylabel('Frequency')
        ax.set_title('Response Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Latency vs Accuracy
        ax = axes[1, 1]
        ax.scatter(df['gemma_latency'], df['gemma_is_correct'], 
                  alpha=0.6, label='Fine-tuned Gemma', s=30, color='#3498db')
        ax.scatter(df['chatgpt_latency'], df['chatgpt_is_correct'], 
                  alpha=0.6, label='ChatGPT', s=30, color='#e74c3c')
        ax.set_xlabel('Latency (seconds)')
        ax.set_ylabel('Correctness (1=correct, 0=incorrect)')
        ax.set_title('Speed vs Accuracy Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Keyword F1 Distribution
        ax = axes[1, 2]
        ax.boxplot([df['gemma_keyword_f1'], df['chatgpt_keyword_f1']],
                   labels=['Fine-tuned Gemma', 'ChatGPT'])
        ax.set_ylabel('Keyword F1 Score')
        ax.set_title('Domain Keyword F1 Distribution')
        ax.grid(True, alpha=0.3)
        
        # BERTScore vs Semantic Similarity
        ax = axes[2, 0]
        ax.scatter(df['gemma_bertscore_f1'], df['gemma_semantic_similarity'], 
                  alpha=0.6, label='Fine-tuned Gemma', s=30, color='#3498db')
        ax.scatter(df['chatgpt_bertscore_f1'], df['chatgpt_semantic_similarity'], 
                  alpha=0.6, label='ChatGPT', s=30, color='#e74c3c')
        ax.set_xlabel('BERTScore F1')
        ax.set_ylabel('Semantic Similarity')
        ax.set_title('BERTScore vs Semantic Similarity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Coherence vs Length
        ax = axes[2, 1]
        ax.scatter(df['gemma_response_length'], df['gemma_coherence'], 
                  alpha=0.6, label='Fine-tuned Gemma', s=30, color='#3498db')
        ax.scatter(df['chatgpt_response_length'], df['chatgpt_coherence'], 
                  alpha=0.6, label='ChatGPT', s=30, color='#e74c3c')
        ax.set_xlabel('Response Length (words)')
        ax.set_ylabel('Coherence Score')
        ax.set_title('Response Length vs Coherence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Overall Quality Score (composite)
        ax = axes[2, 2]
        # Create composite quality score
        gemma_quality = (
            df['gemma_semantic_similarity'] * 0.3 +
            df['gemma_bertscore_f1'] * 0.3 +
            df['gemma_keyword_f1'] * 0.2 +
            df['gemma_coherence'] * 0.2
        )
        chatgpt_quality = (
            df['chatgpt_semantic_similarity'] * 0.3 +
            df['chatgpt_bertscore_f1'] * 0.3 +
            df['chatgpt_keyword_f1'] * 0.2 +
            df['chatgpt_coherence'] * 0.2
        )
        
        ax.boxplot([gemma_quality, chatgpt_quality],
                   labels=['Fine-tuned Gemma', 'ChatGPT'])
        ax.set_ylabel('Composite Quality Score')
        ax.set_title('Overall Quality Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/v3_enhanced_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # V3: Enhanced Radar Chart
        fig2, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        categories = ['Accuracy', 'Semantic Sim', 'Confidence', 'BLEU-4', 'ROUGE-L', 
                     'BERTScore', 'Diversity', 'Coherence', 'Speed', 'Quality']
        
        gemma_values = [
            df['gemma_is_correct'].mean(),
            df['gemma_semantic_similarity'].mean(),
            df['gemma_semantic_confidence'].mean(),
            df['gemma_bleu_4'].mean(),
            df['gemma_rougeL_f'].mean(),
            df['gemma_bertscore_f1'].mean(),
            df['gemma_distinct_2'].mean(),
            df['gemma_coherence'].mean(),
            1 - (df['gemma_latency'].mean() / (df['gemma_latency'].mean() + df['chatgpt_latency'].mean())),
            gemma_quality.mean()
        ]
        
        chatgpt_values = [
            df['chatgpt_is_correct'].mean(),
            df['chatgpt_semantic_similarity'].mean(),
            df['chatgpt_semantic_confidence'].mean(),
            df['chatgpt_bleu_4'].mean(),
            df['chatgpt_rougeL_f'].mean(),
            df['chatgpt_bertscore_f1'].mean(),
            df['chatgpt_distinct_2'].mean(),
            df['chatgpt_coherence'].mean(),
            1 - (df['chatgpt_latency'].mean() / (df['gemma_latency'].mean() + df['chatgpt_latency'].mean())),
            chatgpt_quality.mean()
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        gemma_values += gemma_values[:1]
        chatgpt_values += chatgpt_values[:1]
        angles += angles[:1]
        
        ax.plot(angles, gemma_values, 'o-', linewidth=3, label='Fine-tuned Gemma', color='#3498db')
        ax.fill(angles, gemma_values, alpha=0.25, color='#3498db')
        ax.plot(angles, chatgpt_values, 'o-', linewidth=3, label='ChatGPT', color='#e74c3c')
        ax.fill(angles, chatgpt_values, alpha=0.25, color='#e74c3c')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('V3 Enhanced Comprehensive Metric Comparison', 
                    size=16, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/v3_enhanced_radar_chart_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_advanced_report(self, stats: Dict, df: pd.DataFrame) -> str:
        """V3: Enhanced comprehensive text report"""
        report = []
        report.append("=" * 95)
        report.append("COMPREHENSIVE MODEL COMPARISON REPORT V3")
        report.append("Enhanced Evaluation with Dual Semantic Models & Adaptive Thresholds")
        report.append("=" * 95)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nModels Compared:")
        report.append(f"  1. Fine-tuned Gemma 3 1B (from {self.gemma_path})")
        report.append(f"  2. ChatGPT GPT-4o-mini (OpenAI)")
        report.append(f"\nEvaluation Dataset: {len(self.eval_data)} Q&A pairs")
        report.append(f"Semantic Models: paraphrase-mpnet-base-v2 + all-mpnet-base-v2 (averaged)")
        
        report.append("\n" + "=" * 95)
        report.append("1. V3 ENHANCED PERFORMANCE METRICS")
        report.append("=" * 95)
        
        # Accuracy with confidence
        report.append("\n📊 ACCURACY (Percentage of Correct Answers)")
        report.append("-" * 55)
        report.append(f"Fine-tuned Gemma: {stats['gemma']['accuracy']:.4f} ({stats['gemma']['accuracy']*100:.2f}%)")
        report.append(f"ChatGPT:          {stats['chatgpt']['accuracy']:.4f} ({stats['chatgpt']['accuracy']*100:.2f}%)")
        diff = stats['gemma']['accuracy'] - stats['chatgpt']['accuracy']
        winner = "Fine-tuned Gemma" if diff > 0 else "ChatGPT"
        report.append(f"→ Winner: {winner} (Δ = {abs(diff):.4f})")
        
        # Enhanced semantic similarity
        report.append("\n🔗 V3 ENHANCED SEMANTIC SIMILARITY (Dual Model Average)")
        report.append("-" * 55)
        report.append(f"Fine-tuned Gemma:")
        report.append(f"  Similarity: {stats['gemma']['avg_semantic_similarity']:.4f}")
        report.append(f"  Confidence: {stats['gemma']['avg_semantic_confidence']:.4f}")
        report.append(f"ChatGPT:")
        report.append(f"  Similarity: {stats['chatgpt']['avg_semantic_similarity']:.4f}")
        report.append(f"  Confidence: {stats['chatgpt']['avg_semantic_confidence']:.4f}")
        
        sem_diff = stats['gemma']['avg_semantic_similarity'] - stats['chatgpt']['avg_semantic_similarity']
        winner = "Fine-tuned Gemma" if sem_diff > 0 else "ChatGPT"
        report.append(f"→ Winner: {winner} (Δ = {abs(sem_diff):.4f})")
        
        # Enhanced keyword metrics
        report.append("\n📊 V3 WEIGHTED KEYWORD METRICS (High/Medium Importance)")
        report.append("-" * 55)
        report.append(f"Fine-tuned Gemma:")
        report.append(f"  Precision: {stats['gemma']['precision']:.4f}")
        report.append(f"  Recall:    {stats['gemma']['recall']:.4f}")
        report.append(f"  F1-Score:  {stats['gemma']['f1_score']:.4f}")
        report.append(f"ChatGPT:")
        report.append(f"  Precision: {stats['chatgpt']['precision']:.4f}")
        report.append(f"  Recall:    {stats['chatgpt']['recall']:.4f}")
        report.append(f"  F1-Score:  {stats['chatgpt']['f1_score']:.4f}")
        
        f1_diff = stats['gemma']['f1_score'] - stats['chatgpt']['f1_score']
        winner = "Fine-tuned Gemma" if f1_diff > 0 else "ChatGPT"
        report.append(f"→ Winner (F1): {winner} (Δ = {abs(f1_diff):.4f})")
        
        # Response quality
        report.append("\n📝 V3 RESPONSE QUALITY METRICS")
        report.append("-" * 55)
        report.append(f"Fine-tuned Gemma:")
        report.append(f"  Avg Length:     {stats['gemma']['avg_response_length']:.1f} words")
        report.append(f"  Length Ratio:   {stats['gemma']['avg_length_ratio']:.2f}")
        report.append(f"  Avg Word Len:   {stats['gemma']['avg_word_length']:.2f} chars")
        report.append(f"ChatGPT:")
        report.append(f"  Avg Length:     {stats['chatgpt']['avg_response_length']:.1f} words")
        report.append(f"  Length Ratio:   {stats['chatgpt']['avg_length_ratio']:.2f}")
        report.append(f"  Avg Word Len:   {stats['chatgpt']['avg_word_length']:.2f} chars")
        
        # Enhanced diversity
        report.append("\n🎨 V3 ENHANCED DIVERSITY METRICS")
        report.append("-" * 55)
        report.append(f"Fine-tuned Gemma:")
        report.append(f"  Distinct-1: {stats['gemma']['avg_distinct_1']:.4f}")
        report.append(f"  Distinct-2: {stats['gemma']['avg_distinct_2']:.4f}")
        report.append(f"  Distinct-3: {stats['gemma']['avg_distinct_3']:.4f}")
        report.append(f"ChatGPT:")
        report.append(f"  Distinct-1: {stats['chatgpt']['avg_distinct_1']:.4f}")
        report.append(f"  Distinct-2: {stats['chatgpt']['avg_distinct_2']:.4f}")
        report.append(f"  Distinct-3: {stats['chatgpt']['avg_distinct_3']:.4f}")
        
        # Latency
        report.append("\n⚡ LATENCY (Response Speed)")
        report.append("-" * 55)
        report.append(f"Fine-tuned Gemma: {stats['gemma']['avg_latency']:.3f}s (median: {stats['gemma']['median_latency']:.3f}s)")
        report.append(f"ChatGPT:          {stats['chatgpt']['avg_latency']:.3f}s (median: {stats['chatgpt']['median_latency']:.3f}s)")
        
        latency_diff = stats['gemma']['avg_latency'] - stats['chatgpt']['avg_latency']
        winner = "Fine-tuned Gemma" if latency_diff < 0 else "ChatGPT"
        speedup = abs(latency_diff)
        report.append(f"→ Winner: {winner} ({speedup:.3f}s faster on average)")
        
        # Overall Assessment
        report.append("\n" + "=" * 95)
        report.append("2. V3 OVERALL ASSESSMENT")
        report.append("=" * 95)
        
        # Calculate wins
        wins = {'gemma': 0, 'chatgpt': 0}
        
        categories = [
            ('Accuracy', stats['gemma']['accuracy'] > stats['chatgpt']['accuracy']),
            ('Semantic Similarity', stats['gemma']['avg_semantic_similarity'] > stats['chatgpt']['avg_semantic_similarity']),
            ('Keyword F1', stats['gemma']['f1_score'] > stats['chatgpt']['f1_score']),
            ('BERTScore F1', stats['gemma']['avg_bertscore_f1'] > stats['chatgpt']['avg_bertscore_f1']),
            ('Diversity (Trigrams)', stats['gemma']['avg_distinct_3'] > stats['chatgpt']['avg_distinct_3']),
            ('Coherence', stats['gemma']['avg_coherence'] > stats['chatgpt']['avg_coherence']),
            ('Speed', stats['gemma']['avg_latency'] < stats['chatgpt']['avg_latency']),
            ('Semantic Confidence', stats['gemma']['avg_semantic_confidence'] > stats['chatgpt']['avg_semantic_confidence']),
        ]
        
        report.append("\nV3 Metric-by-Metric Winners:")
        report.append("-" * 55)
        for category, gemma_wins in categories:
            winner_name = "✓ Fine-tuned Gemma" if gemma_wins else "✓ ChatGPT"
            report.append(f"  {category:.<30} {winner_name}")
            if gemma_wins:
                wins['gemma'] += 1
            else:
                wins['chatgpt'] += 1
        
        report.append(f"\n{'='*55}")
        report.append(f"V3 FINAL SCORE:")
        report.append(f"  Fine-tuned Gemma: {wins['gemma']}/8 categories")
        report.append(f"  ChatGPT:          {wins['chatgpt']}/8 categories")
        report.append(f"{'='*55}")
        
        # V3 specific recommendations
        report.append("\n" + "=" * 95)
        report.append("3. V3 ENHANCED RECOMMENDATIONS")
        report.append("=" * 95)
        
        if wins['gemma'] >= 4:
            report.append("\n✅ V3 DEPLOYMENT RECOMMENDATION: Fine-tuned Gemma")
            report.append("-" * 55)
            report.append("Your fine-tuned model demonstrates competitive/superior performance")
            report.append("across multiple enhanced evaluation criteria.")
        else:
            report.append("\n📊 V3 RECOMMENDATION: Continue Enhancement")
            report.append("-" * 55)
            report.append("Consider additional improvements based on V3 analysis.")
        
        # V3 improvements
        report.append("\n" + "=" * 95)
        report.append("4. V3 EVALUATION ENHANCEMENTS")
        report.append("=" * 95)
        report.append("\n✅ V3 NEW FEATURES:")
        report.append("  • Dual semantic similarity models (paraphrase-mpnet + all-mpnet)")
        report.append("  • Semantic confidence scoring based on model agreement")
        report.append("  • Adaptive correctness thresholds based on question complexity")
        report.append("  • Weighted keyword importance (high/medium priority terms)")
        report.append("  • Enhanced response cleaning and preprocessing")
        report.append("  • Trigram diversity analysis (Distinct-3)")
        report.append("  • Repetition penalty in generation")
        report.append("  • Comprehensive warning suppression")
        report.append("  • Response quality metrics (length, word length)")
        report.append("  • Enhanced visualization suite")
        
        report.append("\n📈 V3 IMPROVEMENTS OVER V2:")
        report.append("  • More robust semantic similarity evaluation")
        report.append("  • Better domain-specific keyword analysis")
        report.append("  • Cleaner output with suppressed warnings")
        report.append("  • Enhanced diversity metrics")
        report.append("  • Improved response quality assessment")
        
        report.append("\n" + "=" * 95)
        report.append("END OF V3 COMPREHENSIVE REPORT")
        report.append("=" * 95)
        report.append(f"\nFor detailed per-question results, see: detailed_results_*.csv")
        report.append(f"For V3 enhanced visualizations, see: *.png files in comparison_result_v3/ directory")
        
        return "\n".join(report)


def main():
    """Main execution function for V3"""
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
        print("Initializing Advanced Model Comparison System V3")
        print("Enhanced with Dual Semantic Models & Adaptive Evaluation")
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
        print("✅ V3 Comparison Complete!")
        print("=" * 80)
        print("\nGenerated files in comparison_result_v3/:")
        print("  📊 detailed_results_*.csv - All responses and metrics")
        print("  📈 v3_enhanced_metrics_*.png - V3 performance visualizations")
        print("  📈 v3_enhanced_radar_chart_*.png - V3 comprehensive comparison")
        print("  📝 comprehensive_report_*.txt - V3 detailed analysis")
        print("  📋 statistics_*.json - V3 statistics with metadata")
        print("\n🚀 V3 MAJOR ENHANCEMENTS:")
        print("  ✅ Dual semantic similarity models (paraphrase-mpnet + all-mpnet)")
        print("  ✅ Semantic confidence scoring")
        print("  ✅ Adaptive correctness thresholds")
        print("  ✅ Weighted keyword importance")
        print("  ✅ Enhanced response cleaning")
        print("  ✅ Trigram diversity analysis")
        print("  ✅ Comprehensive warning suppression")
        print("  ✅ Response quality metrics")
        print("  ✅ Enhanced visualization suite")
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