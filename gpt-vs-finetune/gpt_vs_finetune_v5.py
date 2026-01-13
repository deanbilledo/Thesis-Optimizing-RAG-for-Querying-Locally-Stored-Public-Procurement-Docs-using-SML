#!/usr/bin/env python3
"""
Comprehensive Model Comparison: ChatGPT vs Finetuned Model vs Finetuned + RAG
This script evaluates three different approaches:
1. ChatGPT (GPT-4) baseline
2. Finetuned Gemma3 model
3. Finetuned Gemma3 model with RAG enhancement

The script uses:
- gpt_plus_gemma.jsonl for direct GPT/finetuned comparison
- rag_plus_finetune.jsonl for RAG-enhanced evaluation (using input as context)
"""

import json
import os
import time
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Data class to store evaluation results"""
    question: str
    expected_answer: str
    chatgpt_answer: str
    finetune_answer: str
    finetune_rag_answer: str
    context: Optional[str]
    chatgpt_score: float
    finetune_score: float
    finetune_rag_score: float
    response_time_chatgpt: float
    response_time_finetune: float
    response_time_finetune_rag: float

class ChatGPTEvaluator:
    """Handles ChatGPT API calls and evaluations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    async def get_response(self, session: aiohttp.ClientSession, question: str, context: str = None) -> Tuple[str, float]:
        """Get response from ChatGPT with optional context"""
        start_time = time.time()
        
        # Prepare the prompt based on whether context is provided
        if context:
            system_prompt = f"""You are an expert in Philippine government procurement law (RA 9184). 
            Use the following context to answer questions accurately and concisely.
            
            Context: {context}
            
            Answer the question based on the provided context. If the context doesn't contain enough information, 
            state that clearly."""
            user_prompt = question
        else:
            system_prompt = """You are an expert in Philippine government procurement law (RA 9184). 
            Answer questions accurately and concisely based on your knowledge of the law."""
            user_prompt = question
        
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }
        
        try:
            async with session.post(self.base_url, headers=self.headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    answer = data['choices'][0]['message']['content'].strip()
                    response_time = time.time() - start_time
                    return answer, response_time
                else:
                    error_text = await response.text()
                    logger.error(f"ChatGPT API error: {response.status} - {error_text}")
                    return f"Error: {response.status}", time.time() - start_time
        except Exception as e:
            logger.error(f"Exception calling ChatGPT API: {e}")
            return f"Exception: {str(e)}", time.time() - start_time

class FinetuneEvaluator:
    """Handles finetuned model loading and evaluation"""
    
    def __init__(self, model_path: str = "c:/model-training-true/models/gemma3_1b_procurement_weska_v1"):
        self.model_path = model_path
        self.base_model_path = "google/gemma-1.1-2b-it"  # Try the 1.1 version first
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_model(self):
        """Load the finetuned model and tokenizer"""
        try:
            logger.info(f"Loading finetuned model from {self.model_path}")
            
            # Check if this is a LoRA adapter by looking for adapter_config.json
            adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
            
            if os.path.exists(adapter_config_path):
                logger.info("Detected LoRA adapter, loading with PEFT...")
                try:
                    from peft import PeftModel, PeftConfig
                    
                    # Read adapter config to get base model
                    with open(adapter_config_path, 'r') as f:
                        import json
                        adapter_config = json.load(f)
                        base_model_name = adapter_config.get('base_model_name_or_path', 'google/gemma-1.1-2b-it')
                        logger.info(f"Base model from config: {base_model_name}")
                        
                        # Use the base model from config if available
                        if base_model_name and base_model_name != self.base_model_path:
                            self.base_model_path = base_model_name
                    
                    # Try different base models if the first one fails
                    base_models_to_try = [
                        self.base_model_path,
                        "google/gemma-1.1-2b-it",
                        "google/gemma-2b-it", 
                        "google/gemma-1.1-7b-it"
                    ]
                    
                    model_loaded = False
                    for base_model in base_models_to_try:
                        try:
                            logger.info(f"Trying base model: {base_model}")
                            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                            base_model_obj = AutoModelForCausalLM.from_pretrained(
                                base_model,
                                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                                device_map="auto" if self.device.type == "cuda" else None
                            )
                            
                            # Load LoRA adapter
                            logger.info("Loading LoRA adapter...")
                            self.model = PeftModel.from_pretrained(base_model_obj, self.model_path)
                            model_loaded = True
                            logger.info(f"Successfully loaded with base model: {base_model}")
                            break
                            
                        except Exception as e:
                            logger.warning(f"Failed with base model {base_model}: {e}")
                            continue
                    
                    if not model_loaded:
                        raise Exception("Failed to load with any base model")
                        
                except ImportError:
                    logger.warning("PEFT not installed, trying alternative approach...")
                    raise Exception("PEFT required for LoRA adapters")
                    
            else:
                # Try loading as merged model with different base models
                merged_model_paths = [
                    self.model_path,
                    "c:/model-training-true/gemma3_v2_merged_model"
                ]
                
                model_loaded = False
                for model_path in merged_model_paths:
                    try:
                        logger.info(f"Trying merged model path: {model_path}")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_path,
                            trust_remote_code=True
                        )
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                            device_map="auto" if self.device.type == "cuda" else None,
                            trust_remote_code=True
                        )
                        model_loaded = True
                        logger.info(f"Successfully loaded merged model from: {model_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load merged model from {model_path}: {e}")
                        continue
                
                if not model_loaded:
                    logger.info("Falling back to base model...")
                    # Fallback to base model
                    self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        "google/gemma-1.1-2b-it",
                        torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                        device_map="auto" if self.device.type == "cuda" else None
                    )
                    logger.warning("Using base model instead of finetuned model")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_response(self, question: str, context: str = None) -> Tuple[str, float]:
        """Get response from finetuned model"""
        start_time = time.time()
        
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        try:
            # Prepare the prompt
            if context:
                prompt = f"""<bos><start_of_turn>user
Context: {context}

Question: {question}
<end_of_turn>
<start_of_turn>model
"""
            else:
                prompt = f"""<bos><start_of_turn>user
{question}
<end_of_turn>
<start_of_turn>model
"""
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the model's response
            if "<start_of_turn>model\n" in generated_text:
                response = generated_text.split("<start_of_turn>model\n")[-1]
            else:
                response = generated_text[len(prompt):]
            
            response = response.strip()
            response_time = time.time() - start_time
            
            return response, response_time
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}", time.time() - start_time

class SemanticEvaluator:
    """Handles semantic similarity evaluation using sentence transformers"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        try:
            embeddings = self.model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

class ComprehensiveEvaluator:
    """Main evaluator class that orchestrates the comparison"""
    
    def __init__(self, api_key: str):
        self.chatgpt = ChatGPTEvaluator(api_key)
        self.finetune = FinetuneEvaluator()
        self.semantic_eval = SemanticEvaluator()
        self.results = []
        
    def load_test_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load test data from JSONL files"""
        logger.info("Loading test data...")
        
        # Load GPT vs Gemma comparison data (no context)
        gpt_gemma_data = []
        try:
            with open("c:/model-training-true/gpt_plus_gemma.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    gpt_gemma_data.append(data)
        except Exception as e:
            logger.error(f"Error loading gpt_plus_gemma.jsonl: {e}")
            
        # Load RAG + Finetune data (with context)
        rag_finetune_data = []
        try:
            with open("c:/model-training-true/rag_plus_finetune.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line.strip())
                    rag_finetune_data.append(data)
        except Exception as e:
            logger.error(f"Error loading rag_plus_finetune.jsonl: {e}")
            
        logger.info(f"Loaded {len(gpt_gemma_data)} GPT/Gemma samples and {len(rag_finetune_data)} RAG samples")
        return gpt_gemma_data, rag_finetune_data
    
    async def evaluate_sample(self, session: aiohttp.ClientSession, sample: Dict, use_context: bool = False) -> EvaluationResult:
        """Evaluate a single sample across all three approaches"""
        question = sample["instruction"]
        expected_answer = sample["output"]
        context = sample.get("input", "") if use_context else None
        
        print(f"\n{'='*100}")
        print(f"QUESTION: {question}")
        print(f"EXPECTED ANSWER: {expected_answer}")
        if context:
            print(f"CONTEXT PROVIDED: {context[:200]}{'...' if len(context) > 200 else ''}")
        print(f"{'='*100}")
        
        # Get responses from all three approaches
        print("\n🤖 Getting ChatGPT response...")
        chatgpt_answer, chatgpt_time = await self.chatgpt.get_response(session, question, context)
        print(f"ChatGPT Response ({chatgpt_time:.2f}s): {chatgpt_answer}")
        
        print("\n🧠 Getting Finetuned Model response...")
        finetune_answer, finetune_time = self.finetune.get_response(question)
        print(f"Finetuned Response ({finetune_time:.2f}s): {finetune_answer}")
        
        print("\n🔍 Getting Finetuned + RAG response...")
        finetune_rag_answer, finetune_rag_time = self.finetune.get_response(question, context) if context else (finetune_answer, finetune_time)
        if context:
            print(f"Finetuned + RAG Response ({finetune_rag_time:.2f}s): {finetune_rag_answer}")
        else:
            print(f"Finetuned + RAG Response (same as finetune, no context): {finetune_rag_answer}")
        
        # Calculate semantic similarities
        chatgpt_score = self.semantic_eval.calculate_similarity(expected_answer, chatgpt_answer)
        finetune_score = self.semantic_eval.calculate_similarity(expected_answer, finetune_answer)
        finetune_rag_score = self.semantic_eval.calculate_similarity(expected_answer, finetune_rag_answer)
        
        print(f"\n📊 SIMILARITY SCORES:")
        print(f"ChatGPT:        {chatgpt_score:.4f}")
        print(f"Finetuned:      {finetune_score:.4f}")
        print(f"Finetune + RAG: {finetune_rag_score:.4f}")
        
        # Determine winner
        scores = {'ChatGPT': chatgpt_score, 'Finetuned': finetune_score, 'Finetune+RAG': finetune_rag_score}
        winner = max(scores, key=scores.get)
        print(f"🏆 WINNER: {winner} (Score: {scores[winner]:.4f})")
        
        print(f"{'='*100}\n")
        
        return EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            chatgpt_answer=chatgpt_answer,
            finetune_answer=finetune_answer,
            finetune_rag_answer=finetune_rag_answer,
            context=context,
            chatgpt_score=chatgpt_score,
            finetune_score=finetune_score,
            finetune_rag_score=finetune_rag_score,
            response_time_chatgpt=chatgpt_time,
            response_time_finetune=finetune_time,
            response_time_finetune_rag=finetune_rag_time
        )
    
    async def run_evaluation(self, max_samples: int = 50):
        """Run the comprehensive evaluation"""
        logger.info("Starting comprehensive evaluation...")
        
        # Load test data
        gpt_gemma_data, rag_finetune_data = self.load_test_data()
        
        # Take a subset for evaluation
        eval_samples = gpt_gemma_data[:max_samples//2] + rag_finetune_data[:max_samples//2]
        
        print(f"\n🚀 Starting evaluation of {len(eval_samples)} samples...")
        print(f"📊 {len([s for s in eval_samples if s.get('input', '').strip()])} samples with RAG context")
        print(f"📋 {len([s for s in eval_samples if not s.get('input', '').strip()])} samples without context")
        
        async with aiohttp.ClientSession() as session:
            for i, sample in enumerate(eval_samples):
                print(f"\n{'='*20} SAMPLE {i+1}/{len(eval_samples)} {'='*20}")
                
                # For RAG samples, use context; for others, don't
                use_context = "input" in sample and sample["input"].strip()
                result = await self.evaluate_sample(session, sample, use_context)
                self.results.append(result)
                
                # Add small delay to avoid rate limiting
                if i % 5 == 0 and i > 0:
                    print("⏸️  Pausing briefly to respect API rate limits...")
                    await asyncio.sleep(2)
        
        logger.info(f"Completed evaluation of {len(self.results)} samples")
    
    def analyze_results(self) -> Dict:
        """Analyze the evaluation results and generate statistics"""
        if not self.results:
            return {}
        
        # Calculate aggregate statistics
        chatgpt_scores = [r.chatgpt_score for r in self.results]
        finetune_scores = [r.finetune_score for r in self.results]
        finetune_rag_scores = [r.finetune_rag_score for r in self.results]
        
        chatgpt_times = [r.response_time_chatgpt for r in self.results]
        finetune_times = [r.response_time_finetune for r in self.results]
        finetune_rag_times = [r.response_time_finetune_rag for r in self.results]
        
        # Separate results by whether context was used
        context_results = [r for r in self.results if r.context]
        no_context_results = [r for r in self.results if not r.context]
        
        analysis = {
            "overall_stats": {
                "total_samples": len(self.results),
                "samples_with_context": len(context_results),
                "samples_without_context": len(no_context_results),
                "chatgpt_avg_score": np.mean(chatgpt_scores),
                "finetune_avg_score": np.mean(finetune_scores),
                "finetune_rag_avg_score": np.mean(finetune_rag_scores),
                "chatgpt_avg_time": np.mean(chatgpt_times),
                "finetune_avg_time": np.mean(finetune_times),
                "finetune_rag_avg_time": np.mean(finetune_rag_times)
            },
            "performance_comparison": {
                "best_performer": "ChatGPT" if np.mean(chatgpt_scores) > max(np.mean(finetune_scores), np.mean(finetune_rag_scores)) 
                               else "Finetune+RAG" if np.mean(finetune_rag_scores) > np.mean(finetune_scores) 
                               else "Finetune",
                "chatgpt_wins": sum(1 for r in self.results if r.chatgpt_score > max(r.finetune_score, r.finetune_rag_score)),
                "finetune_wins": sum(1 for r in self.results if r.finetune_score > max(r.chatgpt_score, r.finetune_rag_score)),
                "finetune_rag_wins": sum(1 for r in self.results if r.finetune_rag_score > max(r.chatgpt_score, r.finetune_score))
            }
        }
        
        # Context-specific analysis
        if context_results:
            context_chatgpt_scores = [r.chatgpt_score for r in context_results]
            context_finetune_scores = [r.finetune_score for r in context_results]
            context_finetune_rag_scores = [r.finetune_rag_score for r in context_results]
            
            analysis["context_analysis"] = {
                "chatgpt_with_context": np.mean(context_chatgpt_scores),
                "finetune_with_context": np.mean(context_finetune_scores),
                "finetune_rag_with_context": np.mean(context_finetune_rag_scores),
                "rag_improvement": np.mean(context_finetune_rag_scores) - np.mean(context_finetune_scores)
            }
        
        return analysis
    
    def save_results(self, filename: str = None):
        """Save detailed results to CSV, JSON, and HTML"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_evaluation_results_{timestamp}"
        
        # Prepare data for CSV
        csv_data = []
        for r in self.results:
            csv_data.append({
                "question": r.question,
                "expected_answer": r.expected_answer,
                "chatgpt_answer": r.chatgpt_answer,
                "finetune_answer": r.finetune_answer,
                "finetune_rag_answer": r.finetune_rag_answer,
                "context_provided": bool(r.context),
                "context": r.context or "",
                "chatgpt_score": r.chatgpt_score,
                "finetune_score": r.finetune_score,
                "finetune_rag_score": r.finetune_rag_score,
                "chatgpt_time": r.response_time_chatgpt,
                "finetune_time": r.response_time_finetune,
                "finetune_rag_time": r.response_time_finetune_rag,
                "winner": "ChatGPT" if r.chatgpt_score > max(r.finetune_score, r.finetune_rag_score)
                         else "Finetune+RAG" if r.finetune_rag_score > r.finetune_score
                         else "Finetune"
            })
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        csv_path = f"c:/model-training-true/results/{filename}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Save analysis to JSON
        analysis = self.analyze_results()
        json_path = f"c:/model-training-true/results/{filename}_analysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # Save detailed HTML report
        html_path = f"c:/model-training-true/results/{filename}_detailed_report.html"
        self._save_html_report(html_path, analysis)
        
        logger.info(f"Results saved to {csv_path}, {json_path}, and {html_path}")
        return csv_path, json_path, html_path
    
    def _save_html_report(self, filepath: str, analysis: Dict):
        """Save a detailed HTML report with all responses"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Model Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 20px; margin-bottom: 30px; }}
        .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
        .sample {{ border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px; background-color: #fafafa; }}
        .question {{ font-weight: bold; color: #2c3e50; margin-bottom: 10px; }}
        .expected {{ color: #27ae60; margin-bottom: 15px; font-style: italic; }}
        .context {{ color: #7f8c8d; margin-bottom: 15px; background-color: #ecf0f1; padding: 10px; border-radius: 3px; }}
        .response {{ margin: 10px 0; padding: 10px; border-radius: 3px; }}
        .chatgpt {{ background-color: #e8f5e8; border-left: 4px solid #4CAF50; }}
        .finetune {{ background-color: #e8f0ff; border-left: 4px solid #2196F3; }}
        .rag {{ background-color: #fff3e0; border-left: 4px solid #FF9800; }}
        .score {{ font-weight: bold; }}
        .winner {{ background-color: #ffeb3b; padding: 2px 8px; border-radius: 3px; font-weight: bold; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
        .improvement {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Comprehensive Model Evaluation Report</h1>
            <p>ChatGPT vs Finetuned Model vs Finetuned + RAG</p>
            <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="summary">
            <h2>Summary Statistics</h2>
            <div class="stats">
                <div class="stat-box">
                    <h3>Overall Performance</h3>
                    <p>Total Samples: {analysis['overall_stats']['total_samples']}</p>
                    <p>Best Performer: <span class="winner">{analysis['performance_comparison']['best_performer']}</span></p>
                </div>
                <div class="stat-box">
                    <h3>Average Scores</h3>
                    <p>ChatGPT: {analysis['overall_stats']['chatgpt_avg_score']:.4f}</p>
                    <p>Finetuned: {analysis['overall_stats']['finetune_avg_score']:.4f}</p>
                    <p>Finetune+RAG: {analysis['overall_stats']['finetune_rag_avg_score']:.4f}</p>
                </div>
                <div class="stat-box">
                    <h3>Win Counts</h3>
                    <p>ChatGPT: {analysis['performance_comparison']['chatgpt_wins']}</p>
                    <p>Finetuned: {analysis['performance_comparison']['finetune_wins']}</p>
                    <p>Finetune+RAG: {analysis['performance_comparison']['finetune_rag_wins']}</p>
                </div>
                <div class="stat-box">
                    <h3>Response Times (avg)</h3>
                    <p>ChatGPT: {analysis['overall_stats']['chatgpt_avg_time']:.3f}s</p>
                    <p>Finetuned: {analysis['overall_stats']['finetune_avg_time']:.3f}s</p>
                    <p>Finetune+RAG: {analysis['overall_stats']['finetune_rag_avg_time']:.3f}s</p>
                </div>
            </div>
        </div>
        
        <h2>Detailed Sample Results</h2>
"""
        
        # Add each sample
        for i, result in enumerate(self.results, 1):
            scores = {'ChatGPT': result.chatgpt_score, 'Finetuned': result.finetune_score, 'Finetune+RAG': result.finetune_rag_score}
            winner = max(scores, key=scores.get)
            
            rag_improvement = ""
            if result.context and result.finetune_rag_score > result.finetune_score:
                improvement = result.finetune_rag_score - result.finetune_score
                rag_improvement = f'<span class="improvement">RAG Improvement: +{improvement:.4f}</span>'
            
            html_content += f"""
        <div class="sample">
            <h3>Sample {i} - Winner: <span class="winner">{winner}</span> {rag_improvement}</h3>
            
            <div class="question">Question: {result.question}</div>
            <div class="expected">Expected Answer: {result.expected_answer}</div>
            
            {"<div class='context'>Context: " + result.context + "</div>" if result.context else ""}
            
            <div class="response chatgpt">
                <strong>ChatGPT Response</strong> <span class="score">(Score: {result.chatgpt_score:.4f}, Time: {result.response_time_chatgpt:.2f}s)</span><br>
                {result.chatgpt_answer}
            </div>
            
            <div class="response finetune">
                <strong>Finetuned Model Response</strong> <span class="score">(Score: {result.finetune_score:.4f}, Time: {result.response_time_finetune:.2f}s)</span><br>
                {result.finetune_answer}
            </div>
            
            <div class="response rag">
                <strong>Finetuned + RAG Response</strong> <span class="score">(Score: {result.finetune_rag_score:.4f}, Time: {result.response_time_finetune_rag:.2f}s)</span><br>
                {result.finetune_rag_answer}
            </div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

    def print_detailed_results(self, top_n: int = 5):
        """Print detailed results for top N best and worst performing samples"""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "="*120)
        print("DETAILED RESPONSE ANALYSIS")
        print("="*120)
        
        # Sort by overall performance (average of all three scores)
        sorted_results = sorted(self.results, 
                              key=lambda x: (x.chatgpt_score + x.finetune_score + x.finetune_rag_score) / 3, 
                              reverse=True)
        
        print(f"\n🏆 TOP {top_n} BEST PERFORMING SAMPLES:")
        print("="*120)
        for i, result in enumerate(sorted_results[:top_n], 1):
            self._print_sample_detail(result, i, "BEST")
        
        print(f"\n❌ TOP {top_n} WORST PERFORMING SAMPLES:")
        print("="*120)
        for i, result in enumerate(sorted_results[-top_n:], 1):
            self._print_sample_detail(result, i, "WORST")
        
        # Show samples where RAG made the biggest difference
        rag_improvement = [r for r in self.results if r.context and r.finetune_rag_score > r.finetune_score]
        if rag_improvement:
            rag_improvement.sort(key=lambda x: x.finetune_rag_score - x.finetune_score, reverse=True)
            print(f"\n🔍 TOP {min(top_n, len(rag_improvement))} SAMPLES WHERE RAG HELPED MOST:")
            print("="*120)
            for i, result in enumerate(rag_improvement[:top_n], 1):
                improvement = result.finetune_rag_score - result.finetune_score
                print(f"\n{i}. RAG IMPROVEMENT: +{improvement:.4f}")
                self._print_sample_detail(result, i, "RAG_BOOST")
    
    def _print_sample_detail(self, result: EvaluationResult, index: int, category: str):
        """Print detailed information for a single sample"""
        print(f"\n{index}. [{category}]")
        print(f"Question: {result.question}")
        print(f"Expected: {result.expected_answer}")
        
        if result.context:
            print(f"Context: {result.context[:150]}{'...' if len(result.context) > 150 else ''}")
        
        print(f"\n📝 RESPONSES:")
        print(f"ChatGPT ({result.chatgpt_score:.4f}): {result.chatgpt_answer}")
        print(f"Finetuned ({result.finetune_score:.4f}): {result.finetune_answer}")
        print(f"Finetune+RAG ({result.finetune_rag_score:.4f}): {result.finetune_rag_answer}")
        
        scores = {'ChatGPT': result.chatgpt_score, 'Finetuned': result.finetune_score, 'Finetune+RAG': result.finetune_rag_score}
        winner = max(scores, key=scores.get)
        print(f"🏆 Winner: {winner}")
        print("-" * 120)

    def print_summary(self):
        """Print a summary of the evaluation results"""
        analysis = self.analyze_results()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nTotal Samples Evaluated: {analysis['overall_stats']['total_samples']}")
        print(f"Samples with Context (RAG): {analysis['overall_stats']['samples_with_context']}")
        print(f"Samples without Context: {analysis['overall_stats']['samples_without_context']}")
        
        print("\nAVERAGE SEMANTIC SIMILARITY SCORES:")
        print(f"ChatGPT:           {analysis['overall_stats']['chatgpt_avg_score']:.4f}")
        print(f"Finetuned Model:   {analysis['overall_stats']['finetune_avg_score']:.4f}")
        print(f"Finetune + RAG:    {analysis['overall_stats']['finetune_rag_avg_score']:.4f}")
        
        print("\nAVERAGE RESPONSE TIMES (seconds):")
        print(f"ChatGPT:           {analysis['overall_stats']['chatgpt_avg_time']:.3f}")
        print(f"Finetuned Model:   {analysis['overall_stats']['finetune_avg_time']:.3f}")
        print(f"Finetune + RAG:    {analysis['overall_stats']['finetune_rag_avg_time']:.3f}")
        
        print("\nPERFORMANCE COMPARISON:")
        print(f"Best Overall Performer: {analysis['performance_comparison']['best_performer']}")
        print(f"ChatGPT Wins:       {analysis['performance_comparison']['chatgpt_wins']}")
        print(f"Finetuned Wins:     {analysis['performance_comparison']['finetune_wins']}")
        print(f"Finetune+RAG Wins:  {analysis['performance_comparison']['finetune_rag_wins']}")
        
        if "context_analysis" in analysis:
            print("\nCONTEXT-SPECIFIC ANALYSIS:")
            print(f"ChatGPT with Context:     {analysis['context_analysis']['chatgpt_with_context']:.4f}")
            print(f"Finetune with Context:    {analysis['context_analysis']['finetune_with_context']:.4f}")
            print(f"Finetune+RAG with Context: {analysis['context_analysis']['finetune_rag_with_context']:.4f}")
            print(f"RAG Improvement:          {analysis['context_analysis']['rag_improvement']:.4f}")
        
        print("\n" + "="*80)

async def main():
    """Main function to run the comprehensive evaluation"""
    # Configuration
    API_KEY = "REPLACED_FOR_SECURITY"
    MAX_SAMPLES = 100  # Comprehensive evaluation with more samples
    
    print("🤖 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 50)
    print("Comparing: ChatGPT vs Finetuned Gemma vs Finetuned + RAG")
    print(f"Max samples to evaluate: {MAX_SAMPLES}")
    print("=" * 50)
    
    try:
        # Initialize evaluator
        print("\n📋 Initializing evaluator...")
        evaluator = ComprehensiveEvaluator(API_KEY)
        
        # Run evaluation
        print("\n🔄 Running evaluation...")
        await evaluator.run_evaluation(max_samples=MAX_SAMPLES)
        
        # Print summary
        evaluator.print_summary()
        
        # Print detailed results
        evaluator.print_detailed_results(top_n=3)
        
        # Save results
        csv_path, json_path, html_path = evaluator.save_results()
        
        print(f"\nDetailed results saved to:")
        print(f"CSV: {csv_path}")
        print(f"Analysis: {json_path}")
        print(f"HTML Report: {html_path}")
        print(f"\nOpen the HTML file in your browser for a detailed visual report!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("c:/model-training-true/results", exist_ok=True)
    
    # Run the evaluation
    asyncio.run(main())
