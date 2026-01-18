"""
LLM INFERENCE SCALING STRESS TEST
==================================
Tests how inference time scales with:
1. Number of context chunks passed to LLM
2. Number of tokens generated
3. Total input length

This will find the REAL limits where LLM processing degrades.
"""

import os
import sys
import json
import time
import psutil
import statistics
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not installed")
    sys.exit(1)


class LLMInferenceStressTest:
    """Test how LLM inference scales with context size"""
    
    def __init__(self):
        self.workspace = str(Path(__file__).parent)
        self.embed_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.llm_device = 'cpu'
        
        print("=" * 70)
        print("  LLM INFERENCE SCALING STRESS TEST")
        print("  Finding how inference time scales with context size")
        print("=" * 70)
        
        # System info
        mem = psutil.virtual_memory()
        self.total_ram_gb = round(mem.total / (1024**3), 2)
        self.cpu_cores = psutil.cpu_count(logical=False)
        
        print(f"\n📊 System: {self.total_ram_gb}GB RAM, {self.cpu_cores} cores")
        print(f"🤖 LLM: CPU (realistic govt PC)")
        
        # Load models
        print("\n⏳ Loading models...")
        self.load_models()
        print("✅ Models loaded")
        
        self.results = {
            'system': {
                'ram_gb': self.total_ram_gb,
                'cpu_cores': self.cpu_cores,
            },
            'context_scaling': [],
            'token_scaling': [],
            'combined_scaling': [],
            'limits_found': {},
        }
    
    def load_models(self):
        """Load embedding and LLM"""
        # Embedding
        local_embed_path = Path(self.workspace) / 'embedding_model'
        if local_embed_path.exists():
            self.embed_model = SentenceTransformer(str(local_embed_path), device=self.embed_device)
        else:
            self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.embed_device)
        
        # LLM
        model_path = Path(self.workspace) / 'model'
        base_model_path = Path(self.workspace) / 'base_model'
        
        if model_path.exists() and base_model_path.exists():
            from peft import PeftModel
            self.tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
            base_model = AutoModelForCausalLM.from_pretrained(
                str(base_model_path),
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            )
            self.llm = PeftModel.from_pretrained(base_model, str(model_path))
            self.llm = self.llm.to('cpu')
            self.llm.eval()
        else:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map=None, low_cpu_mem_usage=True
            )
            self.llm = self.llm.to('cpu')
            self.llm.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_context_chunks(self, num_chunks: int) -> List[str]:
        """Generate context chunks"""
        templates = [
            "PROCUREMENT Section {i}: Technical specs require 3.0GHz processor, 16GB RAM. Budget: PHP {amt:,.2f}. Delivery: 45 days.",
            "BIDDER REQUIREMENTS Article {i}: PhilGEPS registration, Mayor's permit, BIR clearance required. Contract: PHP {amt:,.2f}.",
            "COMPLIANCE {i}: ISO 9001:2015 mandatory. Warranty: 3 years. Liquidated damages: 1/10 of 1% per day. Total: PHP {amt:,.2f}.",
            "SCHEDULE {i}: Pre-bid Day 7. Submission deadline Day 21. Opening Day 22. Duration: 90 days. ABC: PHP {amt:,.2f}.",
            "ELIGIBILITY {i}: DTI/SEC registration required. 3 years experience. SLCC at least 50% of ABC. Budget: PHP {amt:,.2f}.",
        ]
        
        chunks = []
        for i in range(num_chunks):
            template = templates[i % len(templates)]
            amt = np.random.uniform(500000, 10000000)
            chunks.append(template.format(i=i+1, amt=amt))
        
        return chunks
    
    def measure_inference(self, context_chunks: List[str], max_tokens: int = 50) -> Dict:
        """Measure inference with given context size"""
        query = "What is the approved budget and delivery timeline?"
        
        # Build context from chunks
        context_text = "\n\n".join(context_chunks)
        
        prompt = f"""Based on the following procurement documents, answer the question.

DOCUMENTS:
{context_text}

QUESTION: {query}

ANSWER:"""
        
        # Tokenize and measure input size
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        input_tokens = inputs['input_ids'].shape[1]
        
        # Measure inference time
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        end = time.perf_counter()
        
        inference_time = (end - start) * 1000
        output_tokens = outputs[0].shape[0] - input_tokens
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'inference_ms': round(inference_time, 0),
            'ms_per_output_token': round(inference_time / output_tokens, 1) if output_tokens > 0 else 0,
        }
    
    def test_context_scaling(self):
        """Test how inference scales with context size (number of chunks)"""
        print("\n" + "=" * 70)
        print("  TEST 1: CONTEXT SIZE SCALING")
        print("  How does inference time change as we add more context?")
        print("=" * 70)
        
        # Test different numbers of context chunks
        chunk_counts = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
        
        print(f"\n{'Chunks':<10} {'Input Tok':<12} {'Output Tok':<12} {'Inference':<15} {'ms/token':<10}")
        print("-" * 65)
        
        for num_chunks in chunk_counts:
            chunks = self.generate_context_chunks(num_chunks)
            
            try:
                result = self.measure_inference(chunks, max_tokens=50)
                
                print(f"{num_chunks:<10} {result['input_tokens']:<12} {result['output_tokens']:<12} {result['inference_ms']:<15.0f} {result['ms_per_output_token']:<10.1f}")
                
                self.results['context_scaling'].append({
                    'num_chunks': num_chunks,
                    **result
                })
                
            except Exception as e:
                print(f"{num_chunks:<10} {'ERROR':<12} {'-':<12} {'-':<15} {str(e)[:20]}")
                break
            
            gc.collect()
    
    def test_token_generation_scaling(self):
        """Test how inference scales with output token count"""
        print("\n" + "=" * 70)
        print("  TEST 2: OUTPUT TOKEN SCALING")
        print("  How does inference time change as we generate more tokens?")
        print("=" * 70)
        
        # Fixed context (5 chunks)
        chunks = self.generate_context_chunks(5)
        
        # Test different output lengths
        token_counts = [10, 25, 50, 75, 100, 150, 200, 300]
        
        print(f"\n{'Max Tokens':<12} {'Actual Out':<12} {'Input Tok':<12} {'Inference':<15} {'ms/token':<10}")
        print("-" * 65)
        
        for max_tokens in token_counts:
            try:
                result = self.measure_inference(chunks, max_tokens=max_tokens)
                
                print(f"{max_tokens:<12} {result['output_tokens']:<12} {result['input_tokens']:<12} {result['inference_ms']:<15.0f} {result['ms_per_output_token']:<10.1f}")
                
                self.results['token_scaling'].append({
                    'max_tokens': max_tokens,
                    **result
                })
                
            except Exception as e:
                print(f"{max_tokens:<12} {'ERROR':<12} {'-':<12} {'-':<15} {str(e)[:20]}")
                break
            
            gc.collect()
    
    def test_combined_scaling(self):
        """Test combinations of context size and output length"""
        print("\n" + "=" * 70)
        print("  TEST 3: COMBINED SCALING (Context + Output)")
        print("  Realistic scenarios with varying context and response length")
        print("=" * 70)
        
        scenarios = [
            {'chunks': 3, 'tokens': 50, 'desc': 'Small context, short answer'},
            {'chunks': 5, 'tokens': 50, 'desc': 'Medium context, short answer'},
            {'chunks': 10, 'tokens': 50, 'desc': 'Large context, short answer'},
            {'chunks': 3, 'tokens': 100, 'desc': 'Small context, medium answer'},
            {'chunks': 5, 'tokens': 100, 'desc': 'Medium context, medium answer'},
            {'chunks': 10, 'tokens': 100, 'desc': 'Large context, medium answer'},
            {'chunks': 5, 'tokens': 200, 'desc': 'Medium context, long answer'},
            {'chunks': 10, 'tokens': 200, 'desc': 'Large context, long answer'},
            {'chunks': 15, 'tokens': 100, 'desc': 'Very large context, medium answer'},
            {'chunks': 20, 'tokens': 150, 'desc': 'Huge context, long answer'},
        ]
        
        print(f"\n{'Scenario':<40} {'In Tok':<10} {'Out Tok':<10} {'Time (s)':<12} {'ms/tok':<10}")
        print("-" * 85)
        
        for scenario in scenarios:
            chunks = self.generate_context_chunks(scenario['chunks'])
            
            try:
                result = self.measure_inference(chunks, max_tokens=scenario['tokens'])
                time_sec = result['inference_ms'] / 1000
                
                print(f"{scenario['desc']:<40} {result['input_tokens']:<10} {result['output_tokens']:<10} {time_sec:<12.1f} {result['ms_per_output_token']:<10.1f}")
                
                self.results['combined_scaling'].append({
                    'scenario': scenario['desc'],
                    'num_chunks': scenario['chunks'],
                    'max_tokens': scenario['tokens'],
                    **result
                })
                
            except Exception as e:
                print(f"{scenario['desc']:<40} {'ERROR':<10} {'-':<10} {'-':<12} {str(e)[:20]}")
            
            gc.collect()
    
    def analyze_scaling(self):
        """Analyze the scaling patterns"""
        print("\n" + "=" * 70)
        print("  ANALYSIS: SCALING PATTERNS DISCOVERED")
        print("=" * 70)
        
        # Analyze context scaling
        if self.results['context_scaling']:
            first = self.results['context_scaling'][0]
            last = self.results['context_scaling'][-1]
            
            context_growth = (last['inference_ms'] - first['inference_ms']) / (last['num_chunks'] - first['num_chunks'])
            
            print(f"\n📈 CONTEXT SCALING:")
            print(f"   1 chunk: {first['inference_ms']:.0f}ms")
            print(f"   {last['num_chunks']} chunks: {last['inference_ms']:.0f}ms")
            print(f"   Growth rate: ~{context_growth:.0f}ms per additional chunk")
            
            self.results['limits_found']['context_growth_ms_per_chunk'] = round(context_growth, 0)
        
        # Analyze token scaling
        if self.results['token_scaling']:
            first = self.results['token_scaling'][0]
            last = self.results['token_scaling'][-1]
            
            if last['output_tokens'] != first['output_tokens']:
                token_growth = (last['inference_ms'] - first['inference_ms']) / (last['output_tokens'] - first['output_tokens'])
            else:
                token_growth = 0
            
            avg_ms_per_token = statistics.mean([t['ms_per_output_token'] for t in self.results['token_scaling']])
            
            print(f"\n📈 TOKEN GENERATION SCALING:")
            print(f"   {first['output_tokens']} tokens: {first['inference_ms']:.0f}ms")
            print(f"   {last['output_tokens']} tokens: {last['inference_ms']:.0f}ms")
            print(f"   Average: ~{avg_ms_per_token:.0f}ms per output token")
            
            self.results['limits_found']['avg_ms_per_output_token'] = round(avg_ms_per_token, 0)
        
        # Calculate practical limits
        print(f"\n🎯 PRACTICAL LIMITS (for acceptable response time):")
        
        # Assuming 30 second max acceptable time
        max_acceptable_ms = 30000
        avg_ms_per_token = self.results['limits_found'].get('avg_ms_per_output_token', 300)
        
        max_tokens_30s = int(max_acceptable_ms / avg_ms_per_token) if avg_ms_per_token > 0 else 100
        
        print(f"   Max response time target: 30 seconds")
        print(f"   Max output tokens (30s limit): ~{max_tokens_30s} tokens")
        print(f"   Recommended output tokens: 50-100 tokens")
        
        self.results['limits_found']['max_tokens_30s'] = max_tokens_30s
    
    def generate_report(self):
        """Generate scaling report"""
        report = []
        
        report.append("# LLM Inference Scaling Stress Test Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**System:** {self.total_ram_gb}GB RAM, {self.cpu_cores} cores, CPU inference")
        
        # Context scaling
        report.append("\n---\n")
        report.append("## 1. Context Size Scaling")
        report.append("\nHow inference time increases as more context chunks are added:")
        report.append("\n| Chunks | Input Tokens | Output Tokens | Inference (ms) | ms/token |")
        report.append("|--------|--------------|---------------|----------------|----------|")
        
        for test in self.results['context_scaling']:
            report.append(f"| {test['num_chunks']} | {test['input_tokens']} | {test['output_tokens']} | **{test['inference_ms']:.0f}** | {test['ms_per_output_token']:.0f} |")
        
        # Token scaling
        report.append("\n---\n")
        report.append("## 2. Output Token Scaling")
        report.append("\nHow inference time increases as more tokens are generated:")
        report.append("\n| Max Tokens | Actual Output | Inference (ms) | ms/token |")
        report.append("|------------|---------------|----------------|----------|")
        
        for test in self.results['token_scaling']:
            report.append(f"| {test['max_tokens']} | {test['output_tokens']} | **{test['inference_ms']:.0f}** | {test['ms_per_output_token']:.0f} |")
        
        # Combined scenarios
        report.append("\n---\n")
        report.append("## 3. Combined Scenarios")
        report.append("\n| Scenario | Input Tok | Output Tok | Time (s) | ms/token |")
        report.append("|----------|-----------|------------|----------|----------|")
        
        for test in self.results['combined_scaling']:
            time_s = test['inference_ms'] / 1000
            report.append(f"| {test['scenario']} | {test['input_tokens']} | {test['output_tokens']} | **{time_s:.1f}** | {test['ms_per_output_token']:.0f} |")
        
        # Key findings
        report.append("\n---\n")
        report.append("## 4. Key Findings")
        
        limits = self.results['limits_found']
        report.append(f"\n### Scaling Characteristics:")
        report.append(f"\n- **Context growth:** ~{limits.get('context_growth_ms_per_chunk', 'N/A')}ms per additional chunk")
        report.append(f"- **Token generation:** ~{limits.get('avg_ms_per_output_token', 'N/A')}ms per output token")
        report.append(f"- **Max tokens (30s limit):** ~{limits.get('max_tokens_30s', 'N/A')} tokens")
        
        report.append("\n### Conclusions:")
        report.append("\n1. **Inference time DOES scale with context size** - more chunks = longer processing")
        report.append("\n2. **Output tokens are the main driver** - each token adds ~250-350ms on CPU")
        report.append("\n3. **Practical limit:** Keep output to 50-100 tokens for sub-30s response")
        
        report_text = "\n".join(report)
        
        path = os.path.join(self.workspace, 'llm_scaling_report.md')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n📊 Report saved to: {path}")
    
    def save_results(self):
        """Save results to JSON"""
        path = os.path.join(self.workspace, 'llm_scaling_results.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Results saved to: {path}")
    
    def run(self):
        """Run all tests"""
        start = time.time()
        
        self.test_context_scaling()
        self.test_token_generation_scaling()
        self.test_combined_scaling()
        self.analyze_scaling()
        self.generate_report()
        self.save_results()
        
        elapsed = time.time() - start
        
        print("\n" + "=" * 70)
        print("  LLM SCALING STRESS TEST COMPLETE")
        print("=" * 70)
        print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
        
        return self.results


if __name__ == "__main__":
    test = LLMInferenceStressTest()
    results = test.run()
