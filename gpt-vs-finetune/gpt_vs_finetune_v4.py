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
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

# Evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Toxicity detection
try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    print("Warning: detoxify not installed. Toxicity metrics will be skipped.")
    DETOXIFY_AVAILABLE = False

class SimpleRAG:
    """Simple RAG implementation for context-aware responses"""
    
    def __init__(self, knowledge_base: List[Dict]):
        """
        Initialize RAG with knowledge base
        
        Args:
            knowledge_base: List of dicts with 'instruction', 'input', 'output'
        """
        self.knowledge_base = knowledge_base
        
        # Extract contexts and instructions for retrieval
        self.contexts = []
        self.instructions = []
        self.outputs = []
        
        for item in knowledge_base:
            if item.get('input', '').strip():  # Only use items with input/context
                self.contexts.append(item['input'])
                self.instructions.append(item['instruction'])
                self.outputs.append(item['output'])
        
        print(f"RAG initialized with {len(self.contexts)} knowledge items")
        
        # Initialize TF-IDF vectorizer for retrieval
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            if self.contexts:
                self.context_vectors = self.vectorizer.fit_transform(self.contexts + self.instructions)
            else:
                self.context_vectors = None
    
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context for a query"""
        if not self.contexts or self.context_vectors is None:
            return ""
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                query_vector = self.vectorizer.transform([query])
                
                # Calculate similarities
                similarities = cosine_similarity(query_vector, self.context_vectors).flatten()
                
                # Get top-k most similar contexts
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                retrieved_contexts = []
                for idx in top_indices:
                    if similarities[idx] > 0.1:  # Threshold for relevance
                        if idx < len(self.contexts):
                            retrieved_contexts.append(self.contexts[idx])
                        else:
                            # If index is beyond contexts, it's from instructions
                            inst_idx = idx - len(self.contexts)
                            if inst_idx < len(self.instructions):
                                retrieved_contexts.append(f"Q: {self.instructions[inst_idx]}\nA: {self.outputs[inst_idx]}")
                
                return "\n\n".join(retrieved_contexts[:2])  # Limit to top 2 to avoid token limits
        
        except Exception as e:
            print(f"Warning: RAG retrieval failed: {e}")
            return ""

class TripleModelComparison:
    def __init__(self, gemma_path: str, openai_api_key: str, eval_file: str, knowledge_file: str):
        """
        Initialize triple model comparison: ChatGPT vs Fine-tuned vs Fine-tuned+RAG
        
        Args:
            gemma_path: Path to fine-tuned Gemma model
            openai_api_key: OpenAI API key
            eval_file: Path to eval_qa_pairs.jsonl
            knowledge_file: Path to train_clean.jsonl for RAG knowledge base
        """
        self.gemma_path = gemma_path
        self.eval_file = eval_file
        self.knowledge_file = knowledge_file
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.gpt_model = "gpt-4o-mini"
        
        # Load evaluation data (JSONL format)
        self.eval_data = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.eval_data.append(json.loads(line.strip()))
        
        # Load knowledge base and filter for RAG
        print("Loading knowledge base for RAG...")
        self.knowledge_base = []
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    # Only include items with instruction, input, and output
                    if all(key in item for key in ['instruction', 'input', 'output']):
                        if item['input'].strip() and item['output'].strip():  # Must have non-empty input and output
                            self.knowledge_base.append(item)
        
        print(f"Loaded {len(self.knowledge_base)} knowledge base items with context")
        
        # Initialize RAG system
        self.rag = SimpleRAG(self.knowledge_base)
        
        print("Initializing evaluation models...")
        
        # Initialize semantic models with warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.semantic_model1 = SentenceTransformer('paraphrase-mpnet-base-v2')
            self.semantic_model2 = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize toxicity detector
        if DETOXIFY_AVAILABLE:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.toxicity_model = Detoxify('original')
        else:
            self.toxicity_model = None
        
        # Load Gemma model
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
        
        # Ensure model is in eval mode
        self.gemma_model.eval()
        
        # Check device placement
        if hasattr(self.gemma_model, 'parameters'):
            device = next(self.gemma_model.parameters()).device
            print(f"Model is on device: {device}")
        
        # Clear initial GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.results = []
    
    def get_gemma_response(self, question: str, context: str = "") -> Tuple[str, float]:
        """Get response from fine-tuned Gemma model"""
        if context:
            prompt = f"<start_of_turn>user\nContext: {context}\n\nQuestion: {question}<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = self.gemma_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.gemma_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.gemma_tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True,
            )
        latency = time.time() - start_time
        
        response = self.gemma_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<start_of_turn>model\n")[-1].strip()
        
        # Clean response
        response = self._clean_response(response)
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response, latency
    
    def get_chatgpt_response(self, question: str) -> Tuple[str, float]:
        """Get response from ChatGPT (no context)"""
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
    
    def get_gemma_rag_response(self, question: str) -> Tuple[str, float, str]:
        """Get response from fine-tuned Gemma model with RAG context"""
        # Retrieve relevant context
        context = self.rag.retrieve_context(question)
        
        # Get response with context
        response, latency = self.get_gemma_response(question, context)
        
        return response, latency, context
    
    def _clean_response(self, response: str) -> str:
        """Clean and normalize response text"""
        # Remove common artifacts
        patterns_to_remove = [
            r'<[^>]+>',  # HTML tags
            r'\[.*?\]',  # Brackets
            r'^\s*-+\s*',  # Leading dashes
            r'\s+',  # Multiple whitespace
        ]
        
        cleaned = response
        for pattern in patterns_to_remove[:-1]:  # Don't apply whitespace pattern yet
            cleaned = re.sub(pattern, '', cleaned)
        
        # Apply whitespace normalization last
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def calculate_dual_semantic_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """Calculate semantic similarity using dual models"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Model 1: paraphrase-mpnet-base-v2
                emb1_1, emb1_2 = self.semantic_model1.encode([text1, text2])
                sim1 = cosine_similarity([emb1_1], [emb1_2])[0][0]
                
                # Model 2: all-mpnet-base-v2
                emb2_1, emb2_2 = self.semantic_model2.encode([text1, text2])
                sim2 = cosine_similarity([emb2_1], [emb2_2])[0][0]
                
                # Average and confidence
                avg_similarity = (sim1 + sim2) / 2
                confidence = 1 - abs(sim1 - sim2)  # Higher when models agree
                
                return {
                    'semantic_similarity': float(avg_similarity),
                    'semantic_confidence': float(confidence),
                    'sem_model1': float(sim1),
                    'sem_model2': float(sim2)
                }
        except Exception as e:
            print(f"Warning: Semantic similarity calculation failed: {e}")
            return {
                'semantic_similarity': 0.0,
                'semantic_confidence': 0.0,
                'sem_model1': 0.0,
                'sem_model2': 0.0
            }
    
    def calculate_bleu_score(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate BLEU scores with improved tokenization"""
        try:
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
        except Exception as e:
            return {f'bleu_{n}': 0.0 for n in range(1, 5)}
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = self.rouge_scorer.score(reference, generated)
            
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge1_p': scores['rouge1'].precision,
                'rouge1_r': scores['rouge1'].recall,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rougeL_f': scores['rougeL'].fmeasure,
            }
        except Exception as e:
            return {
                'rouge1_f': 0.0, 'rouge1_p': 0.0, 'rouge1_r': 0.0,
                'rouge2_f': 0.0, 'rougeL_f': 0.0
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
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0,
            }
    
    def calculate_diversity_metrics(self, text: str) -> Dict[str, float]:
        """Calculate enhanced diversity metrics"""
        tokens = text.lower().split()
        
        if len(tokens) == 0:
            return {
                'distinct_1': 0.0, 'distinct_2': 0.0, 'distinct_3': 0.0,
                'lexical_diversity': 0.0, 'avg_word_length': 0.0
            }
        
        # Distinct-1, 2, 3
        distinct_1 = len(set(tokens)) / len(tokens)
        
        bigrams = [' '.join(tokens[i:i+2]) for i in range(len(tokens)-1)]
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        
        trigrams = [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
        distinct_3 = len(set(trigrams)) / len(trigrams) if trigrams else 0.0
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in tokens])
        
        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'distinct_3': distinct_3,
            'lexical_diversity': distinct_1,
            'avg_word_length': float(avg_word_length)
        }
    
    def assess_correctness(self, generated: str, reference: str) -> Dict[str, float]:
        """Enhanced correctness assessment"""
        # Semantic similarity
        sem_scores = self.calculate_dual_semantic_similarity(generated, reference)
        semantic_sim = sem_scores['semantic_similarity']
        
        # BERT F1
        bert_scores = self.calculate_bert_score(generated, reference)
        bert_f1 = bert_scores['bertscore_f1']
        
        # Enhanced keyword analysis
        high_priority_keywords = [
            'RA 9184', 'Government Procurement Reform Act', 'GPRA', 'procurement',
            'bidding', 'BAC', 'GPPB', 'competitive', 'transparency'
        ]
        
        medium_priority_keywords = [
            'contract', 'evaluation', 'technical', 'financial', 'eligibility',
            'award', 'notice', 'specifications', 'compliance', 'supplier',
            'public', 'government', 'sealed', 'envelope', 'IRR'
        ]
        
        gen_lower = generated.lower()
        ref_lower = reference.lower()
        
        # High priority keyword scoring
        gen_high = set(kw.lower() for kw in high_priority_keywords if kw.lower() in gen_lower)
        ref_high = set(kw.lower() for kw in high_priority_keywords if kw.lower() in ref_lower)
        
        # Medium priority keyword scoring
        gen_med = set(kw.lower() for kw in medium_priority_keywords if kw.lower() in gen_lower)
        ref_med = set(kw.lower() for kw in medium_priority_keywords if kw.lower() in ref_lower)
        
        # Calculate weighted keyword scores
        if len(ref_high) > 0 or len(ref_med) > 0:
            high_intersection = gen_high.intersection(ref_high)
            med_intersection = gen_med.intersection(ref_med)
            
            high_precision = len(high_intersection) / len(gen_high) if len(gen_high) > 0 else 0
            high_recall = len(high_intersection) / len(ref_high) if len(ref_high) > 0 else 0
            
            med_precision = len(med_intersection) / len(gen_med) if len(gen_med) > 0 else 0
            med_recall = len(med_intersection) / len(ref_med) if len(ref_med) > 0 else 0
            
            # Weighted keyword F1 (high priority = 70%, medium = 30%)
            weighted_precision = 0.7 * high_precision + 0.3 * med_precision
            weighted_recall = 0.7 * high_recall + 0.3 * med_recall
            
            keyword_f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0
        else:
            weighted_precision = weighted_recall = keyword_f1 = 0
        
        # Multi-criteria correctness with adaptive threshold
        base_score = (
            semantic_sim * 0.35 +          # 35% semantic similarity
            bert_f1 * 0.40 +               # 40% BERT F1
            keyword_f1 * 0.25              # 25% weighted keyword overlap
        )
        
        # Adaptive threshold based on question complexity
        question_length = len(reference.split())
        if question_length < 20:
            threshold = 0.70  # Higher threshold for simple questions
        elif question_length < 50:
            threshold = 0.65  # Medium threshold
        else:
            threshold = 0.60  # Lower threshold for complex questions
        
        is_correct = base_score > threshold
        
        return {
            'is_correct': is_correct,
            'correctness_score': base_score,
            'semantic_similarity': semantic_sim,
            'semantic_confidence': sem_scores['semantic_confidence'],
            'keyword_precision': weighted_precision,
            'keyword_recall': weighted_recall,
            'keyword_f1': keyword_f1,
            'threshold_used': threshold
        }
    
    def calculate_comprehensive_metrics(self, generated: str, reference: str, 
                                       latency: float, context_used: str = "") -> Dict:
        """Calculate all metrics for a single response"""
        metrics = {}
        
        # Basic info
        metrics['latency'] = latency
        metrics['response_length'] = len(generated.split())
        metrics['context_length'] = len(context_used.split()) if context_used else 0
        
        # Correctness assessment
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
        
        return metrics
    
    def run_comparison(self):
        """Run triple model comparison"""
        print(f"\n{'='*80}")
        print(f"TRIPLE MODEL COMPARISON V4")
        print(f"{'='*80}")
        print(f"Model 1: ChatGPT (gpt-4o-mini) - No Context")
        print(f"Model 2: Fine-tuned Gemma - No Context")
        print(f"Model 3: Fine-tuned Gemma + RAG - With Retrieved Context")
        print(f"Evaluation Dataset: {len(self.eval_data)} questions")
        print(f"RAG Knowledge Base: {len(self.knowledge_base)} items")
        print(f"{'='*80}\n")
        
        for idx, qa_pair in enumerate(self.eval_data):
            question = qa_pair['question']
            reference = qa_pair['answer']
            
            print(f"\nProcessing [{idx+1}/{len(self.eval_data)}]: {question[:60]}...")
            
            # Get ChatGPT response (no context)
            chatgpt_response, chatgpt_latency = self.get_chatgpt_response(question)
            time.sleep(0.5)
            
            # Get Fine-tuned Gemma response (no context)
            gemma_response, gemma_latency = self.get_gemma_response(question)
            time.sleep(0.5)
            
            # Get Fine-tuned Gemma + RAG response (with context)
            gemma_rag_response, gemma_rag_latency, rag_context = self.get_gemma_rag_response(question)
            time.sleep(0.5)
            
            # Calculate metrics for all three models
            chatgpt_metrics = self.calculate_comprehensive_metrics(
                chatgpt_response, reference, chatgpt_latency
            )
            
            gemma_metrics = self.calculate_comprehensive_metrics(
                gemma_response, reference, gemma_latency
            )
            
            gemma_rag_metrics = self.calculate_comprehensive_metrics(
                gemma_rag_response, reference, gemma_rag_latency, rag_context
            )
            
            # Store results
            result = {
                'question_id': idx + 1,
                'question': question,
                'reference_answer': reference,
                'chatgpt_response': chatgpt_response,
                'gemma_response': gemma_response,
                'gemma_rag_response': gemma_rag_response,
                'rag_context': rag_context,
            }
            
            # Add metrics with prefixes
            for key, value in chatgpt_metrics.items():
                result[f'chatgpt_{key}'] = value
            
            for key, value in gemma_metrics.items():
                result[f'gemma_{key}'] = value
            
            for key, value in gemma_rag_metrics.items():
                result[f'gemma_rag_{key}'] = value
            
            self.results.append(result)
            
            # Print summary
            print(f"  ChatGPT      - Semantic: {chatgpt_metrics['semantic_similarity']:.3f}±{chatgpt_metrics['semantic_confidence']:.3f} | "
                  f"BERT F1: {chatgpt_metrics['bertscore_f1']:.3f} | "
                  f"Correct: {'✓' if chatgpt_metrics['is_correct'] else '✗'} | "
                  f"Latency: {chatgpt_latency:.2f}s")
            
            print(f"  Gemma        - Semantic: {gemma_metrics['semantic_similarity']:.3f}±{gemma_metrics['semantic_confidence']:.3f} | "
                  f"BERT F1: {gemma_metrics['bertscore_f1']:.3f} | "
                  f"Correct: {'✓' if gemma_metrics['is_correct'] else '✗'} | "
                  f"Latency: {gemma_latency:.2f}s")
            
            print(f"  Gemma+RAG    - Semantic: {gemma_rag_metrics['semantic_similarity']:.3f}±{gemma_rag_metrics['semantic_confidence']:.3f} | "
                  f"BERT F1: {gemma_rag_metrics['bertscore_f1']:.3f} | "
                  f"Correct: {'✓' if gemma_rag_metrics['is_correct'] else '✗'} | "
                  f"Context: {len(rag_context.split())//10*10}+ words | "
                  f"Latency: {gemma_rag_latency:.2f}s")
        
        print(f"\n{'='*80}")
        print("Triple Model Comparison Complete!")
        print(f"{'='*80}\n")
    
    def calculate_aggregate_metrics(self, df: pd.DataFrame, model_prefix: str) -> Dict:
        """Calculate aggregate metrics for a model"""
        metrics = {}
        
        # Core metrics
        metrics['accuracy'] = float(df[f'{model_prefix}_is_correct'].mean())
        metrics['avg_correctness_score'] = float(df[f'{model_prefix}_correctness_score'].mean())
        metrics['precision'] = float(df[f'{model_prefix}_keyword_precision'].mean())
        metrics['recall'] = float(df[f'{model_prefix}_keyword_recall'].mean())
        metrics['f1_score'] = float(df[f'{model_prefix}_keyword_f1'].mean())
        
        # Latency and response metrics
        metrics['avg_latency'] = float(df[f'{model_prefix}_latency'].mean())
        metrics['median_latency'] = float(df[f'{model_prefix}_latency'].median())
        metrics['avg_response_length'] = float(df[f'{model_prefix}_response_length'].mean())
        
        # Semantic metrics
        metrics['avg_semantic_similarity'] = float(df[f'{model_prefix}_semantic_similarity'].mean())
        metrics['avg_semantic_confidence'] = float(df[f'{model_prefix}_semantic_confidence'].mean())
        
        # BLEU and ROUGE
        for n in range(1, 5):
            metrics[f'avg_bleu_{n}'] = float(df[f'{model_prefix}_bleu_{n}'].mean())
        
        metrics['avg_rouge1_f'] = float(df[f'{model_prefix}_rouge1_f'].mean())
        metrics['avg_rouge2_f'] = float(df[f'{model_prefix}_rouge2_f'].mean())
        metrics['avg_rougeL_f'] = float(df[f'{model_prefix}_rougeL_f'].mean())
        
        # BERTScore
        metrics['avg_bertscore_f1'] = float(df[f'{model_prefix}_bertscore_f1'].mean())
        metrics['avg_bertscore_precision'] = float(df[f'{model_prefix}_bertscore_precision'].mean())
        metrics['avg_bertscore_recall'] = float(df[f'{model_prefix}_bertscore_recall'].mean())
        
        # Diversity
        metrics['avg_distinct_1'] = float(df[f'{model_prefix}_distinct_1'].mean())
        metrics['avg_distinct_2'] = float(df[f'{model_prefix}_distinct_2'].mean())
        metrics['avg_distinct_3'] = float(df[f'{model_prefix}_distinct_3'].mean())
        metrics['avg_lexical_diversity'] = float(df[f'{model_prefix}_lexical_diversity'].mean())
        
        # RAG-specific metrics
        if f'{model_prefix}_context_length' in df.columns:
            context_lengths = df[f'{model_prefix}_context_length']
            metrics['avg_context_length'] = float(context_lengths.mean())
            metrics['context_usage_rate'] = float((context_lengths > 0).mean())
        
        return metrics
    
    def generate_report(self, output_dir: str = "comparison_result_v4"):
        """Generate comprehensive triple model comparison report"""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame
        df = pd.DataFrame(self.results)
        
        # Calculate aggregate statistics for all three models
        chatgpt_stats = self.calculate_aggregate_metrics(df, 'chatgpt')
        gemma_stats = self.calculate_aggregate_metrics(df, 'gemma')
        gemma_rag_stats = self.calculate_aggregate_metrics(df, 'gemma_rag')
        
        stats = {
            'chatgpt': chatgpt_stats,
            'gemma': gemma_stats,
            'gemma_rag': gemma_rag_stats,
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'total_questions': len(self.eval_data),
                'knowledge_base_size': len(self.knowledge_base),
                'models_compared': 3,
                'version': 'V4_Triple_Comparison'
            }
        }
        
        # Generate visualizations
        self._create_triple_visualizations(df, output_dir, timestamp)
        
        # Save detailed results
        df.to_csv(f"{output_dir}/detailed_results_{timestamp}.csv", index=False)
        
        # Save statistics
        with open(f"{output_dir}/statistics_{timestamp}.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate text report
        report = self._generate_triple_report(stats, df)
        with open(f"{output_dir}/comprehensive_report_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_dir}/")
        print(f"{'='*80}\n")
        print(report)
        
        return stats
    
    def _create_triple_visualizations(self, df: pd.DataFrame, output_dir: str, timestamp: str):
        """Create visualizations for triple model comparison"""
        sns.set_style("whitegrid")
        
        # Figure 1: Core Performance Metrics (3 models)
        fig1, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig1.suptitle('Triple Model Comparison - Core Performance Metrics', fontsize=16, fontweight='bold')
        
        models = ['chatgpt', 'gemma', 'gemma_rag']
        model_names = ['ChatGPT', 'Gemma', 'Gemma+RAG']
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        # Accuracy
        ax = axes[0, 0]
        accuracy_data = [df[f'{model}_is_correct'].mean() for model in models]
        bars = ax.bar(model_names, accuracy_data, color=colors, alpha=0.7)
        ax.set_ylabel('Accuracy Score')
        ax.set_title('Accuracy Comparison')
        ax.set_ylim([0, 1])
        for bar, acc in zip(bars, accuracy_data):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
        
        # Semantic Similarity
        ax = axes[0, 1]
        sem_sim_data = [df[f'{model}_semantic_similarity'].values for model in models]
        ax.boxplot(sem_sim_data, labels=model_names)
        ax.set_ylabel('Semantic Similarity')
        ax.set_title('Semantic Similarity Distribution')
        ax.grid(True, alpha=0.3)
        
        # BERT F1 Score
        ax = axes[0, 2]
        bert_f1_data = [df[f'{model}_bertscore_f1'].values for model in models]
        ax.boxplot(bert_f1_data, labels=model_names)
        ax.set_ylabel('BERT F1 Score')
        ax.set_title('BERT F1 Score Distribution')
        ax.grid(True, alpha=0.3)
        
        # Latency
        ax = axes[1, 0]
        latency_data = [df[f'{model}_latency'].values for model in models]
        ax.boxplot(latency_data, labels=model_names)
        ax.set_ylabel('Latency (seconds)')
        ax.set_title('Response Latency Distribution')
        ax.grid(True, alpha=0.3)
        
        # F1 Score (Keyword-based)
        ax = axes[1, 1]
        f1_data = [df[f'{model}_keyword_f1'].mean() for model in models]
        bars = ax.bar(model_names, f1_data, color=colors, alpha=0.7)
        ax.set_ylabel('F1 Score')
        ax.set_title('Keyword F1 Score')
        for bar, f1 in zip(bars, f1_data):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                   f'{f1:.3f}', ha='center', va='bottom')
        
        # Response Length
        ax = axes[1, 2]
        length_data = [df[f'{model}_response_length'].values for model in models]
        ax.boxplot(length_data, labels=model_names)
        ax.set_ylabel('Response Length (words)')
        ax.set_title('Response Length Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/triple_core_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: RAG Analysis
        fig2, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig2.suptitle('RAG Performance Analysis', fontsize=16, fontweight='bold')
        
        # Context Usage
        ax = axes[0, 0]
        context_lengths = df['gemma_rag_context_length'].values
        ax.hist(context_lengths, bins=20, alpha=0.7, color='#2ecc71')
        ax.set_xlabel('Context Length (words)')
        ax.set_ylabel('Frequency')
        ax.set_title('RAG Context Length Distribution')
        ax.grid(True, alpha=0.3)
        
        # RAG Impact on Accuracy
        ax = axes[0, 1]
        gemma_acc = df['gemma_is_correct'].values
        gemma_rag_acc = df['gemma_rag_is_correct'].values
        improvement = gemma_rag_acc.astype(int) - gemma_acc.astype(int)
        
        unique, counts = np.unique(improvement, return_counts=True)
        labels = ['RAG Worse', 'No Change', 'RAG Better']
        colors_impact = ['#e74c3c', '#95a5a6', '#2ecc71']
        ax.bar(labels, counts, color=colors_impact, alpha=0.7)
        ax.set_ylabel('Number of Questions')
        ax.set_title('RAG Impact on Accuracy')
        
        # Semantic Similarity: Gemma vs Gemma+RAG
        ax = axes[1, 0]
        ax.scatter(df['gemma_semantic_similarity'], df['gemma_rag_semantic_similarity'], 
                  alpha=0.6, color='#2ecc71')
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.8)
        ax.set_xlabel('Gemma Semantic Similarity')
        ax.set_ylabel('Gemma+RAG Semantic Similarity')
        ax.set_title('Semantic Similarity: Gemma vs Gemma+RAG')
        ax.grid(True, alpha=0.3)
        
        # Context Length vs Performance
        ax = axes[1, 1]
        ax.scatter(df['gemma_rag_context_length'], df['gemma_rag_bertscore_f1'], 
                  alpha=0.6, color='#3498db')
        ax.set_xlabel('Context Length (words)')
        ax.set_ylabel('BERT F1 Score')
        ax.set_title('Context Length vs Performance')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rag_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Triple Model Radar Chart
        fig3, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        categories = ['Accuracy', 'Semantic Sim', 'BERT F1', 'BLEU-4', 'ROUGE-L', 
                     'Precision', 'Recall', 'F1-Score', 'Diversity', 'Speed']
        
        # Calculate values for each model
        model_values = {}
        for i, model in enumerate(models):
            values = [
                df[f'{model}_is_correct'].mean(),
                df[f'{model}_semantic_similarity'].mean(),
                df[f'{model}_bertscore_f1'].mean(),
                df[f'{model}_bleu_4'].mean(),
                df[f'{model}_rougeL_f'].mean(),
                df[f'{model}_keyword_precision'].mean(),
                df[f'{model}_keyword_recall'].mean(),
                df[f'{model}_keyword_f1'].mean(),
                df[f'{model}_distinct_1'].mean(),
                1 - (df[f'{model}_latency'].mean() / df[f'{model}_latency'].max())  # Speed (inverted latency)
            ]
            model_values[model] = values
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, (model, values) in enumerate(model_values.items()):
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model_names[i], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Triple Model Comparison - Comprehensive Metrics', 
                    size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/triple_radar_chart_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_triple_report(self, stats: Dict, df: pd.DataFrame) -> str:
        """Generate comprehensive text report for triple model comparison"""
        report = []
        report.append("=" * 100)
        report.append("COMPREHENSIVE TRIPLE MODEL COMPARISON REPORT V4")
        report.append("ChatGPT vs Fine-tuned Gemma vs Fine-tuned Gemma + RAG")
        report.append("=" * 100)
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nModels Compared:")
        report.append(f"  1. ChatGPT GPT-4o-mini (OpenAI) - No Context")
        report.append(f"  2. Fine-tuned Gemma 3 1B - No Context")
        report.append(f"  3. Fine-tuned Gemma 3 1B + RAG - With Retrieved Context")
        report.append(f"\nEvaluation Dataset: {len(self.eval_data)} Q&A pairs")
        report.append(f"RAG Knowledge Base: {len(self.knowledge_base)} items with context")
        
        report.append("\n" + "=" * 100)
        report.append("1. ACCURACY AND CORRECTNESS METRICS")
        report.append("=" * 100)
        
        # Accuracy comparison
        report.append("\n📊 ACCURACY (Percentage of Correct Answers)")
        report.append("-" * 60)
        report.append(f"ChatGPT:          {stats['chatgpt']['accuracy']:.4f} ({stats['chatgpt']['accuracy']*100:.2f}%)")
        report.append(f"Gemma:            {stats['gemma']['accuracy']:.4f} ({stats['gemma']['accuracy']*100:.2f}%)")
        report.append(f"Gemma + RAG:      {stats['gemma_rag']['accuracy']:.4f} ({stats['gemma_rag']['accuracy']*100:.2f}%)")
        
        # Find winner
        accuracies = [
            ('ChatGPT', stats['chatgpt']['accuracy']),
            ('Gemma', stats['gemma']['accuracy']),
            ('Gemma + RAG', stats['gemma_rag']['accuracy'])
        ]
        winner = max(accuracies, key=lambda x: x[1])
        report.append(f"\n→ Winner: {winner[0]} ({winner[1]*100:.2f}%)")
        
        # RAG Impact
        rag_improvement = stats['gemma_rag']['accuracy'] - stats['gemma']['accuracy']
        report.append(f"\n📈 RAG Impact: {rag_improvement:+.4f} ({rag_improvement*100:+.2f}% points)")
        if rag_improvement > 0:
            report.append("   ✅ RAG provides improvement over base fine-tuned model")
        else:
            report.append("   ⚠️  RAG does not improve over base fine-tuned model")
        
        # Precision, Recall, F1
        report.append("\n📊 PRECISION, RECALL, F1-SCORE (Keyword-based)")
        report.append("-" * 60)
        for model_name, model_key in [('ChatGPT', 'chatgpt'), ('Gemma', 'gemma'), ('Gemma + RAG', 'gemma_rag')]:
            report.append(f"{model_name}:")
            report.append(f"  Precision: {stats[model_key]['precision']:.4f}")
            report.append(f"  Recall:    {stats[model_key]['recall']:.4f}")
            report.append(f"  F1-Score:  {stats[model_key]['f1_score']:.4f}")
            report.append("")
        
        # Semantic Similarity
        report.append("🔗 SEMANTIC SIMILARITY (Higher is Better)")
        report.append("-" * 60)
        report.append(f"ChatGPT:          {stats['chatgpt']['avg_semantic_similarity']:.4f}")
        report.append(f"Gemma:            {stats['gemma']['avg_semantic_similarity']:.4f}")
        report.append(f"Gemma + RAG:      {stats['gemma_rag']['avg_semantic_similarity']:.4f}")
        
        sem_winner = max([
            ('ChatGPT', stats['chatgpt']['avg_semantic_similarity']),
            ('Gemma', stats['gemma']['avg_semantic_similarity']),
            ('Gemma + RAG', stats['gemma_rag']['avg_semantic_similarity'])
        ], key=lambda x: x[1])
        report.append(f"→ Winner: {sem_winner[0]} ({sem_winner[1]:.4f})")
        
        report.append("\n" + "=" * 100)
        report.append("2. PERFORMANCE AND EFFICIENCY METRICS")
        report.append("=" * 100)
        
        # Latency
        report.append("\n⚡ LATENCY (Response Speed - Lower is Better)")
        report.append("-" * 60)
        report.append(f"ChatGPT:          {stats['chatgpt']['avg_latency']:.3f}s")
        report.append(f"Gemma:            {stats['gemma']['avg_latency']:.3f}s")
        report.append(f"Gemma + RAG:      {stats['gemma_rag']['avg_latency']:.3f}s")
        
        latency_winner = min([
            ('ChatGPT', stats['chatgpt']['avg_latency']),
            ('Gemma', stats['gemma']['avg_latency']),
            ('Gemma + RAG', stats['gemma_rag']['avg_latency'])
        ], key=lambda x: x[1])
        report.append(f"→ Fastest: {latency_winner[0]} ({latency_winner[1]:.3f}s)")
        
        # RAG overhead
        rag_overhead = stats['gemma_rag']['avg_latency'] - stats['gemma']['avg_latency']
        report.append(f"\n📊 RAG Overhead: {rag_overhead:+.3f}s ({rag_overhead/stats['gemma']['avg_latency']*100:+.1f}%)")
        
        report.append("\n" + "=" * 100)
        report.append("3. RAG ANALYSIS")
        report.append("=" * 100)
        
        # RAG statistics
        report.append(f"\n📚 RAG Knowledge Base Statistics:")
        report.append(f"  Total Knowledge Items: {len(self.knowledge_base)}")
        report.append(f"  Average Context Length: {stats['gemma_rag']['avg_context_length']:.1f} words")
        report.append(f"  Context Usage Rate: {stats['gemma_rag']['context_usage_rate']*100:.1f}%")
        
        # RAG vs Base Model Comparison
        report.append(f"\n🔄 RAG vs Base Model Comparison:")
        metrics_to_compare = [
            ('Accuracy', 'accuracy'),
            ('Semantic Similarity', 'avg_semantic_similarity'),
            ('BERT F1', 'avg_bertscore_f1'),
            ('Keyword F1', 'f1_score')
        ]
        
        rag_wins = 0
        for metric_name, metric_key in metrics_to_compare:
            gemma_val = stats['gemma'][metric_key]
            rag_val = stats['gemma_rag'][metric_key]
            diff = rag_val - gemma_val
            winner = "RAG" if diff > 0 else "Base"
            if diff > 0:
                rag_wins += 1
            
            report.append(f"  {metric_name:.<25} RAG: {rag_val:.4f} | Base: {gemma_val:.4f} | Δ: {diff:+.4f} | Winner: {winner}")
        
        report.append(f"\n→ RAG wins in {rag_wins}/{len(metrics_to_compare)} metrics")
        
        report.append("\n" + "=" * 100)
        report.append("4. OVERALL RANKING")
        report.append("=" * 100)
        
        # Calculate overall scores
        key_metrics = ['accuracy', 'avg_semantic_similarity', 'avg_bertscore_f1', 'f1_score']
        overall_scores = {}
        
        for model_name, model_key in [('ChatGPT', 'chatgpt'), ('Gemma', 'gemma'), ('Gemma + RAG', 'gemma_rag')]:
            score = np.mean([stats[model_key][metric] for metric in key_metrics])
            overall_scores[model_name] = score
        
        # Sort by score
        ranked_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        report.append("\n🏆 OVERALL RANKING (Based on Key Metrics):")
        report.append("-" * 60)
        for i, (model, score) in enumerate(ranked_models, 1):
            report.append(f"{i}. {model:.<20} {score:.4f}")
        
        report.append("\n" + "=" * 100)
        report.append("5. RECOMMENDATIONS")
        report.append("=" * 100)
        
        winner_model = ranked_models[0][0]
        
        if winner_model == "Gemma + RAG":
            report.append("\n✅ RECOMMENDATION: Deploy Fine-tuned Gemma + RAG")
            report.append("-" * 60)
            report.append("The RAG-enhanced model shows the best overall performance.")
            report.append("\nBenefits:")
            report.append("  • Enhanced accuracy through contextual information")
            report.append("  • Better semantic understanding")
            report.append("  • Maintains local deployment advantages")
            report.append("\nConsiderations:")
            report.append(f"  • Additional latency overhead: {rag_overhead:.2f}s")
            report.append("  • Requires maintaining knowledge base")
        
        elif winner_model == "ChatGPT":
            report.append("\n📊 RECOMMENDATION: Consider ChatGPT for Production")
            report.append("-" * 60)
            report.append("ChatGPT shows superior performance across most metrics.")
            report.append("\nConsiderations for fine-tuned model:")
            report.append("  • Continue improving training data")
            report.append("  • Enhance RAG retrieval system")
            report.append("  • Consider model size or architecture changes")
        
        else:
            report.append("\n⚖️  RECOMMENDATION: Base Fine-tuned Model")
            report.append("-" * 60)
            report.append("The base fine-tuned model provides good performance.")
            report.append("RAG may need improvements in retrieval or context selection.")
        
        report.append("\n" + "=" * 100)
        report.append("6. TECHNICAL IMPROVEMENTS IN V4")
        report.append("=" * 100)
        report.append("\n✅ NEW FEATURES:")
        report.append("  • Three-model comparison: ChatGPT vs Gemma vs Gemma+RAG")
        report.append("  • Simple RAG implementation with TF-IDF retrieval")
        report.append("  • Dual semantic similarity models for confidence scoring")
        report.append("  • Enhanced keyword analysis with weighted priorities")
        report.append("  • Adaptive correctness thresholds based on question complexity")
        report.append("  • RAG-specific metrics and analysis")
        report.append("  • Context usage tracking and visualization")
        report.append("  • Comprehensive three-way performance comparison")
        
        report.append("\n" + "=" * 100)
        report.append("END OF TRIPLE MODEL COMPARISON REPORT V4")
        report.append("=" * 100)
        report.append(f"\nFor detailed per-question results, see: detailed_results_*.csv")
        report.append(f"For visualizations, see: *.png files in comparison_result_v4/ directory")
        
        return "\n".join(report)


def main():
    """Main execution function"""
    # Configuration
    GEMMA_MODEL_PATH = "gemma3_1b_procurement_weska_v2"
    EVAL_FILE = "eval_qa_pairs.jsonl"
    KNOWLEDGE_FILE = "train_clean.jsonl"
    OPENAI_API_KEY = "REPLACED_FOR_SECURITY"
    
    if not OPENAI_API_KEY:
        print("=" * 80)
        print("ERROR: OpenAI API Key Not Found")
        print("=" * 80)
        return
    
    try:
        # Initialize comparison
        print("\n" + "=" * 80)
        print("Initializing Triple Model Comparison System V4")
        print("=" * 80)
        
        comparison = TripleModelComparison(
            gemma_path=GEMMA_MODEL_PATH,
            openai_api_key=OPENAI_API_KEY,
            eval_file=EVAL_FILE,
            knowledge_file=KNOWLEDGE_FILE
        )
        
        # Run comparison
        comparison.run_comparison()
        
        # Generate comprehensive report
        stats = comparison.generate_report()
        
        print("\n" + "=" * 80)
        print("✅ Triple Model Comparison Complete!")
        print("=" * 80)
        print("\nGenerated files in comparison_result_v4/:")
        print("  📊 detailed_results_*.csv - All responses and metrics")
        print("  📈 triple_core_metrics_*.png - Performance comparison")
        print("  📈 rag_analysis_*.png - RAG-specific analysis") 
        print("  📈 triple_radar_chart_*.png - Comprehensive comparison")
        print("  📝 comprehensive_report_*.txt - Full analysis")
        print("  📋 statistics_*.json - Raw statistics")
        print("\n🚀 V4 FEATURES:")
        print("  ✅ Triple model comparison (ChatGPT vs Gemma vs Gemma+RAG)")
        print("  ✅ Simple RAG implementation with TF-IDF retrieval")
        print("  ✅ Context-aware evaluation and analysis")
        print("  ✅ RAG impact analysis and visualization")
        print("  ✅ Enhanced performance metrics")
        print("=" * 80 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure:")
        print(f"  1. Model folder exists: {GEMMA_MODEL_PATH}")
        print(f"  2. Evaluation file exists: {EVAL_FILE}")
        print(f"  3. Knowledge base file exists: {KNOWLEDGE_FILE}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()