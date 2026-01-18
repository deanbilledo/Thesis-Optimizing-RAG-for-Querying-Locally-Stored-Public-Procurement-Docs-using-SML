"""
RAG System CPU STRESS TEST
==========================
Stress test using only CPU - generates documents with varying pages and file sizes.
Tests both retrieval and LLM inference performance limits.
"""

import os
import sys
import json
import time
import psutil
import gc
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not installed")
    sys.exit(1)


class CPUStressTest:
    """CPU-only stress test with varying document sizes"""
    
    # Performance thresholds (milliseconds)
    EXCELLENT_RETRIEVAL = 50
    GOOD_RETRIEVAL = 100
    ACCEPTABLE_RETRIEVAL = 200
    
    def __init__(self):
        self.workspace = str(Path(__file__).parent.parent)
        
        print("=" * 70)
        print("  RAG SYSTEM CPU STRESS TEST")
        print("  Testing with varying document pages and sizes")
        print("=" * 70)
        
        # System info
        mem = psutil.virtual_memory()
        self.total_ram_gb = round(mem.total / (1024**3), 2)
        self.available_ram_gb = round(mem.available / (1024**3), 2)
        self.cpu_cores = psutil.cpu_count(logical=False)
        self.cpu_threads = psutil.cpu_count(logical=True)
        
        print(f"\n{'='*50}")
        print("SYSTEM CONFIGURATION")
        print(f"{'='*50}")
        print(f"  Total RAM: {self.total_ram_gb} GB")
        print(f"  Available RAM: {self.available_ram_gb} GB")
        print(f"  CPU Cores: {self.cpu_cores} physical, {self.cpu_threads} logical")
        print(f"  Device: CPU ONLY (forced)")
        print(f"{'='*50}")
        
        # Load embedding model (CPU)
        print("\n[1/2] Loading embedding model on CPU...")
        embed_path = Path(self.workspace) / 'embedding_model'
        if embed_path.exists():
            self.embed_model = SentenceTransformer(str(embed_path), device='cpu')
        else:
            self.embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        print("      Embedding model loaded on CPU")
        
        # Load LLM (CPU)
        print("[2/2] Loading LLM on CPU...")
        self.load_llm()
        print("      LLM loaded on CPU")
        
        self.results = {
            'test_info': {
                'name': 'CPU Stress Test',
                'timestamp': datetime.now().isoformat(),
                'device': 'CPU',
            },
            'system': {
                'ram_gb': self.total_ram_gb,
                'available_ram_gb': self.available_ram_gb,
                'cpu_cores': self.cpu_cores,
                'cpu_threads': self.cpu_threads,
            },
            'document_tests': [],
            'summary': {},
        }
    
    def load_llm(self):
        """Load LLM on CPU"""
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
            self.model_name = "Gemma-3-1B + LoRA"
        else:
            # Fallback
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            )
            self.llm = self.llm.to('cpu')
            self.llm.eval()
            self.model_name = "TinyLlama-1.1B"
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_document_content(self, num_pages: int, target_size_kb: int) -> Tuple[List[str], int]:
        """
        Generate document chunks simulating a multi-page document.
        
        Args:
            num_pages: Number of pages to simulate
            target_size_kb: Target file size in KB
            
        Returns:
            List of chunks and actual size in KB
        """
        # Approximate: 2.5 chunks per page, ~500 chars per chunk
        chunks_per_page = 2.5
        num_chunks = int(num_pages * chunks_per_page)
        
        # Calculate chars needed for target size (1 char ≈ 1 byte)
        target_chars = target_size_kb * 1024
        chars_per_chunk = target_chars // num_chunks if num_chunks > 0 else 500
        chars_per_chunk = max(300, min(chars_per_chunk, 2000))  # Reasonable limits
        
        templates = [
            """PROCUREMENT DOCUMENT - Section {i}
            
Technical Specifications and Requirements:
The procuring entity requires the following technical specifications for the equipment/services 
being procured. All items must meet or exceed the minimum specifications listed below.

Processor: Minimum 3.0GHz multi-core processor with at least {cores} cores
Memory: {ram}GB DDR4 RAM, expandable to {max_ram}GB
Storage: {storage}GB SSD with read speeds of at least 500MB/s
Display: {display}-inch Full HD (1920x1080) resolution

Approved Budget for the Contract (ABC): PHP {amt:,.2f}
Delivery Period: {days} calendar days from receipt of Notice to Proceed
Warranty: {warranty} years comprehensive on-site warranty

All equipment must be brand new, unused, and of recent manufacture (not older than 1 year from 
date of delivery). The supplier must provide complete documentation including user manuals, 
warranty cards, and certificates of authenticity.""",

            """BIDDER ELIGIBILITY REQUIREMENTS - Article {i}

To be eligible to participate in this procurement, bidders must submit the following documents:

1. PhilGEPS Registration Certificate (Platinum Membership preferred)
2. Valid Mayor's/Business Permit for the current year
3. BIR Tax Clearance Certificate
4. Audited Financial Statements for the last {years} years
5. DTI/SEC Registration Certificate
6. PCAB License (for infrastructure projects)
7. Omnibus Sworn Statement

Single Largest Completed Contract (SLCC):
The bidder must have completed at least one contract similar in nature to this procurement with 
a value of at least fifty percent (50%) of the ABC, which is PHP {slcc:,.2f}.

Net Financial Contracting Capacity (NFCC):
The prospective bidder must have an NFCC of at least equal to the ABC, computed as:
NFCC = [(Current Assets - Current Liabilities) x K] - Value of Outstanding Works

Total Contract Value: PHP {amt:,.2f}
Number of Lots: {lots}""",

            """SCHEDULE OF ACTIVITIES AND TIMELINE - Item {i}

The procurement process shall follow the following schedule in accordance with RA 9184:

Day 1-7: Advertisement and posting of Invitation to Bid
Day 7: Pre-Bid Conference at BAC Conference Room, 10:00 AM
Day 8-20: Preparation of Bids
Day 21: Deadline for Submission of Bids, 10:00 AM sharp
Day 21: Opening of Bids, 10:30 AM
Day 22-28: Bid Evaluation
Day 29-35: Post-Qualification of Lowest Calculated Bid
Day 36: BAC Resolution recommending award
Day 37-42: Approval by Head of Procuring Entity
Day 43: Issuance of Notice of Award
Day 44-50: Contract Preparation and Signing
Day 51: Issuance of Notice to Proceed

Project Duration: {duration} calendar days
Liquidated Damages: 1/10 of 1% of contract price per calendar day of delay
Maximum Deduction: 10% of contract price

Approved Budget: PHP {amt:,.2f}""",

            """TERMS AND CONDITIONS - Clause {i}

PAYMENT TERMS:
Progress billing shall be based on actual deliveries/accomplishments, subject to verification 
by the end-user and inspection by the Technical Working Group. Payment shall be made within 
{payment_days} calendar days from receipt of complete billing documents.

Initial Delivery: {initial}%
Second Delivery: {second}%  
Final Delivery: {final}%

PERFORMANCE SECURITY:
The winning bidder shall post a Performance Security within ten (10) calendar days from 
receipt of the Notice of Award, equivalent to the following percentages of the total 
contract price:

- Cash or Manager's Check: 5%
- Bank Guarantee: 5%
- Surety Bond: 30%

RETENTION MONEY:
A retention money equivalent to {retention}% of every progress payment shall be withheld 
by the procuring entity and shall be released only after final acceptance.

Contract Amount: PHP {amt:,.2f}""",

            """COMPLIANCE AND QUALITY ASSURANCE - Section {i}

All goods, equipment, and services procured must comply with the following standards:

ISO 9001:2015 Quality Management System
ISO 14001:2015 Environmental Management System  
ISO 45001:2018 Occupational Health and Safety
Philippine National Standards (PNS)

TESTING AND INSPECTION:
The Technical Working Group shall conduct inspection and acceptance testing of all deliverables.
The supplier must provide:
- Factory Acceptance Test (FAT) results
- Site Acceptance Test (SAT) upon installation
- Commissioning report
- Training for {trainees} end-users

WARRANTY REQUIREMENTS:
- Parts warranty: {parts_warranty} years
- Labor warranty: {labor_warranty} years
- On-site support response time: {response} hours

Defective items shall be replaced within {replace} calendar days at no additional cost.

Total Project Value: PHP {amt:,.2f}""",
        ]
        
        chunks = []
        total_chars = 0
        
        for i in range(num_chunks):
            template = templates[i % len(templates)]
            
            # Random values for template
            params = {
                'i': i + 1,
                'amt': np.random.uniform(500000, 50000000),
                'slcc': np.random.uniform(250000, 25000000),
                'cores': np.random.choice([4, 6, 8, 12]),
                'ram': np.random.choice([8, 16, 32]),
                'max_ram': np.random.choice([32, 64, 128]),
                'storage': np.random.choice([256, 512, 1000]),
                'display': np.random.choice([14, 15.6, 17.3, 24, 27]),
                'days': np.random.choice([30, 45, 60, 90]),
                'warranty': np.random.choice([1, 2, 3]),
                'years': np.random.choice([2, 3, 5]),
                'lots': np.random.choice([1, 2, 3, 5]),
                'duration': np.random.choice([30, 60, 90, 120, 180]),
                'payment_days': np.random.choice([15, 30, 45]),
                'initial': np.random.choice([30, 40, 50]),
                'second': np.random.choice([30, 35, 40]),
                'final': np.random.choice([20, 25, 30]),
                'retention': np.random.choice([5, 10]),
                'trainees': np.random.choice([5, 10, 15, 20]),
                'parts_warranty': np.random.choice([2, 3, 5]),
                'labor_warranty': np.random.choice([1, 2, 3]),
                'response': np.random.choice([4, 8, 24, 48]),
                'replace': np.random.choice([7, 14, 30]),
            }
            
            chunk = template.format(**params)
            
            # Pad or trim to approximate target size per chunk
            if len(chunk) < chars_per_chunk:
                # Add more content
                padding = f"\n\nAdditional Reference ID: DOC-{i:04d}-{np.random.randint(1000, 9999)}\n"
                padding += f"Cross-reference: Section {np.random.randint(1, 50)}, Article {np.random.randint(1, 100)}\n"
                padding += "Notes: " + "This document is for official use. " * ((chars_per_chunk - len(chunk)) // 40)
                chunk += padding[:chars_per_chunk - len(chunk)]
            
            chunks.append(chunk[:chars_per_chunk])
            total_chars += len(chunks[-1])
        
        actual_size_kb = total_chars // 1024
        return chunks, actual_size_kb
    
    def setup_collection(self, chunks: List[str]) -> Tuple:
        """Create ChromaDB collection with document chunks"""
        # Encode chunks
        embeddings = self.embed_model.encode(chunks, show_progress_bar=False)
        
        # Create collection
        client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
        coll_name = f"stress_{len(chunks)}_{int(time.time()*1000)}"
        collection = client.create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})
        
        # Add in batches
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            collection.add(
                documents=chunks[i:end],
                embeddings=embeddings[i:end].tolist(),
                metadatas=[{"id": j, "chunk_idx": j} for j in range(i, end)],
                ids=[f"chunk_{j}" for j in range(i, end)]
            )
        
        return collection, client
    
    def measure_retrieval(self, collection, num_queries: int = 5) -> Dict:
        """Measure retrieval performance"""
        queries = [
            "What is the approved budget for the contract?",
            "What are the technical specifications required?",
            "What documents are needed for bidder eligibility?",
            "What is the delivery timeline?",
            "What are the warranty requirements?",
        ]
        
        # Warm up
        q_emb = self.embed_model.encode(queries[0])
        _ = collection.query(query_embeddings=[q_emb.tolist()], n_results=5)
        
        # Measure multiple runs
        times = []
        for q in queries[:num_queries]:
            for _ in range(3):  # 3 runs per query
                start = time.perf_counter()
                q_emb = self.embed_model.encode(q)
                _ = collection.query(query_embeddings=[q_emb.tolist()], n_results=5)
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        return {
            'min_ms': round(min(times), 2),
            'max_ms': round(max(times), 2),
            'avg_ms': round(statistics.mean(times), 2),
            'median_ms': round(statistics.median(times), 2),
            'std_ms': round(statistics.stdev(times), 2) if len(times) > 1 else 0,
        }
    
    def measure_inference(self, context: str, query: str) -> Dict:
        """Measure LLM inference performance"""
        prompt = f"""Based on the following procurement document context, answer the question.

Context:
{context[:2000]}

Question: {query}

Answer:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        end = time.perf_counter()
        
        inference_time = (end - start) * 1000
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        
        return {
            'inference_ms': round(inference_time, 2),
            'inference_sec': round(inference_time / 1000, 2),
            'tokens_generated': tokens_generated,
            'tokens_per_sec': round(tokens_generated / (inference_time / 1000), 2) if inference_time > 0 else 0,
        }
    
    def run_test(self, num_pages: int, target_size_kb: int) -> Dict:
        """Run a single stress test with specified document size"""
        print(f"\n  Testing: {num_pages} pages, ~{target_size_kb}KB target size...")
        
        mem_before = psutil.Process().memory_info().rss / (1024**2)
        
        # Generate document
        gen_start = time.perf_counter()
        chunks, actual_size_kb = self.generate_document_content(num_pages, target_size_kb)
        gen_time = (time.perf_counter() - gen_start) * 1000
        
        # Setup ChromaDB
        setup_start = time.perf_counter()
        collection, client = self.setup_collection(chunks)
        setup_time = (time.perf_counter() - setup_start) * 1000
        
        # Measure retrieval
        retrieval_stats = self.measure_retrieval(collection)
        
        # Get context for inference test
        q_emb = self.embed_model.encode("What is the approved budget?")
        results = collection.query(query_embeddings=[q_emb.tolist()], n_results=3)
        context = "\n".join(results['documents'][0]) if results['documents'] else ""
        
        # Measure inference
        inference_stats = self.measure_inference(context, "What is the approved budget for the contract?")
        
        mem_after = psutil.Process().memory_info().rss / (1024**2)
        
        # Cleanup
        del collection
        del client
        gc.collect()
        
        # Classify retrieval performance
        avg_retrieval = retrieval_stats['avg_ms']
        if avg_retrieval < self.EXCELLENT_RETRIEVAL:
            retrieval_rating = "EXCELLENT"
        elif avg_retrieval < self.GOOD_RETRIEVAL:
            retrieval_rating = "GOOD"
        elif avg_retrieval < self.ACCEPTABLE_RETRIEVAL:
            retrieval_rating = "ACCEPTABLE"
        else:
            retrieval_rating = "SLOW"
        
        result = {
            'pages': num_pages,
            'target_size_kb': target_size_kb,
            'actual_size_kb': actual_size_kb,
            'num_chunks': len(chunks),
            'generation_ms': round(gen_time, 2),
            'indexing_ms': round(setup_time, 2),
            'retrieval': retrieval_stats,
            'retrieval_rating': retrieval_rating,
            'inference': inference_stats,
            'memory_used_mb': round(mem_after - mem_before, 2),
            'total_time_sec': round((gen_time + setup_time + retrieval_stats['avg_ms'] + inference_stats['inference_ms']) / 1000, 2),
        }
        
        return result
    
    def run_stress_tests(self):
        """Run all stress tests with varying document sizes"""
        print("\n" + "=" * 70)
        print("  STARTING CPU STRESS TESTS")
        print("=" * 70)
        
        # Test configurations: (pages, target_size_kb)
        test_configs = [
            # Small documents
            (5, 50),      # 5 pages, ~50KB
            (10, 100),    # 10 pages, ~100KB
            
            # Medium documents
            (25, 250),    # 25 pages, ~250KB
            (50, 500),    # 50 pages, ~500KB
            
            # Large documents  
            (100, 1000),  # 100 pages, ~1MB
            (150, 1500),  # 150 pages, ~1.5MB
            
            # Very large documents
            (200, 2000),  # 200 pages, ~2MB
            (300, 3000),  # 300 pages, ~3MB
            
            # Extreme tests (may be slow)
            (500, 5000),  # 500 pages, ~5MB
        ]
        
        print(f"\n{'Pages':<8} {'Size(KB)':<10} {'Chunks':<10} {'Retrieval':<12} {'Inference':<12} {'Memory':<10} {'Status'}")
        print("-" * 75)
        
        for num_pages, target_kb in test_configs:
            # Check available memory - lower threshold since models are already loaded
            mem = psutil.virtual_memory()
            if mem.available < 0.5 * 1024**3:  # Less than 500MB available
                print(f"\n  Warning: Low memory ({mem.available / 1024**3:.1f}GB). Stopping tests.")
                break
            
            try:
                result = self.run_test(num_pages, target_kb)
                self.results['document_tests'].append(result)
                
                # Display result
                status_icon = {
                    'EXCELLENT': '🟢',
                    'GOOD': '🟢',
                    'ACCEPTABLE': '🟡',
                    'SLOW': '🔴'
                }.get(result['retrieval_rating'], '⚪')
                
                print(f"{result['pages']:<8} {result['actual_size_kb']:<10} {result['num_chunks']:<10} "
                      f"{result['retrieval']['avg_ms']:<12.1f} {result['inference']['inference_sec']:<12.2f}s "
                      f"{result['memory_used_mb']:<10.0f}MB {status_icon} {result['retrieval_rating']}")
                
                # Force garbage collection after each test
                gc.collect()
                time.sleep(0.5)  # Brief pause to let memory settle
                
            except Exception as e:
                print(f"{num_pages:<8} {target_kb:<10} {'ERROR':<10} {str(e)[:40]}")
                self.results['document_tests'].append({
                    'pages': num_pages,
                    'target_size_kb': target_kb,
                    'error': str(e),
                })
                break
        
        # Calculate summary
        successful_tests = [t for t in self.results['document_tests'] if 'error' not in t]
        if successful_tests:
            self.results['summary'] = {
                'total_tests': len(successful_tests),
                'max_pages_tested': max(t['pages'] for t in successful_tests),
                'max_size_kb_tested': max(t['actual_size_kb'] for t in successful_tests),
                'avg_retrieval_ms': round(statistics.mean(t['retrieval']['avg_ms'] for t in successful_tests), 2),
                'avg_inference_sec': round(statistics.mean(t['inference']['inference_sec'] for t in successful_tests), 2),
                'model_used': self.model_name,
            }
        
        return self.results
    
    def generate_report(self):
        """Generate markdown report"""
        report = []
        report.append("# RAG System CPU Stress Test Report\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**Device:** CPU Only\n\n")
        
        # System info
        report.append("## System Configuration\n")
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Total RAM | {self.results['system']['ram_gb']} GB |")
        report.append(f"| Available RAM | {self.results['system']['available_ram_gb']} GB |")
        report.append(f"| CPU Cores | {self.results['system']['cpu_cores']} physical |")
        report.append(f"| CPU Threads | {self.results['system']['cpu_threads']} logical |")
        report.append(f"| LLM Model | {self.model_name} |")
        report.append("")
        
        # Test results table
        report.append("## Stress Test Results\n")
        report.append("| Pages | Size (KB) | Chunks | Retrieval (ms) | Inference (s) | Memory (MB) | Rating |")
        report.append("|-------|-----------|--------|----------------|---------------|-------------|--------|")
        
        for test in self.results['document_tests']:
            if 'error' in test:
                report.append(f"| {test['pages']} | {test['target_size_kb']} | - | ERROR | - | - | - |")
            else:
                report.append(f"| {test['pages']} | {test['actual_size_kb']} | {test['num_chunks']} | "
                            f"{test['retrieval']['avg_ms']:.1f} | {test['inference']['inference_sec']:.2f} | "
                            f"{test['memory_used_mb']:.0f} | {test['retrieval_rating']} |")
        
        report.append("")
        
        # Summary
        if self.results.get('summary'):
            report.append("## Summary\n")
            s = self.results['summary']
            report.append(f"- **Total Tests Completed:** {s['total_tests']}")
            report.append(f"- **Maximum Pages Tested:** {s['max_pages_tested']}")
            report.append(f"- **Maximum Size Tested:** {s['max_size_kb_tested']} KB")
            report.append(f"- **Average Retrieval Time:** {s['avg_retrieval_ms']} ms")
            report.append(f"- **Average Inference Time:** {s['avg_inference_sec']} seconds")
            report.append(f"- **Model:** {s['model_used']}")
            report.append("")
        
        # Performance thresholds
        report.append("## Performance Thresholds\n")
        report.append("| Rating | Retrieval Time |")
        report.append("|--------|----------------|")
        report.append(f"| EXCELLENT | < {self.EXCELLENT_RETRIEVAL} ms |")
        report.append(f"| GOOD | < {self.GOOD_RETRIEVAL} ms |")
        report.append(f"| ACCEPTABLE | < {self.ACCEPTABLE_RETRIEVAL} ms |")
        report.append(f"| SLOW | >= {self.ACCEPTABLE_RETRIEVAL} ms |")
        report.append("")
        
        # Notes
        report.append("## Notes\n")
        report.append("- All tests run with CPU only (no GPU acceleration)")
        report.append("- Document content simulates Philippine procurement documents")
        report.append("- Retrieval uses ChromaDB with cosine similarity")
        report.append("- Inference uses the fine-tuned Gemma-3-1B model with LoRA adapter")
        report.append("- Times include embedding generation, vector search, and LLM generation")
        
        return "\n".join(report)


def main():
    print("\n" + "=" * 70)
    print("  RAG SYSTEM CPU STRESS TEST")
    print("  Testing with varying document pages and file sizes")
    print("=" * 70 + "\n")
    
    tester = CPUStressTest()
    results = tester.run_stress_tests()
    
    # Save results
    workspace = Path(__file__).parent.parent
    reports_dir = workspace / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Save JSON
    json_path = reports_dir / 'cpu_stress_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to: {json_path}")
    
    # Save report
    report = tester.generate_report()
    report_path = reports_dir / 'cpu_stress_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Report saved to: {report_path}")
    
    # Print summary
    if results.get('summary'):
        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)
        s = results['summary']
        print(f"  Tests Completed: {s['total_tests']}")
        print(f"  Max Pages: {s['max_pages_tested']}")
        print(f"  Max Size: {s['max_size_kb_tested']} KB")
        print(f"  Avg Retrieval: {s['avg_retrieval_ms']} ms")
        print(f"  Avg Inference: {s['avg_inference_sec']} seconds")
        print("=" * 70)


if __name__ == "__main__":
    main()
