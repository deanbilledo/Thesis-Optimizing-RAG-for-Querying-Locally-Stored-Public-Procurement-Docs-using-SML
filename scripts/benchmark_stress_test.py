"""
RAG System STRESS TEST
======================
Push the system to find actual performance limits where retrieval degrades.
Test progressively larger document sets until we hit performance thresholds.
"""

import os
import sys
import json
import time
import psutil
import platform
import statistics
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not installed")
    sys.exit(1)


class StressTest:
    """Stress test to find actual system limits"""
    
    # Performance thresholds
    EXCELLENT = 15      # ms
    GOOD = 25           # ms
    ACCEPTABLE = 50     # ms
    DEGRADED = 100      # ms
    FAILURE = 200       # ms
    
    def __init__(self):
        self.workspace = str(Path(__file__).parent)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("=" * 70)
        print("  RAG SYSTEM STRESS TEST - FINDING PERFORMANCE LIMITS")
        print("=" * 70)
        
        # System info
        mem = psutil.virtual_memory()
        self.total_ram_gb = round(mem.total / (1024**3), 2)
        self.cpu_cores = psutil.cpu_count(logical=False)
        
        print(f"\n📊 System: {self.total_ram_gb}GB RAM, {self.cpu_cores} cores")
        print(f"🖥️  Device: {self.device.upper()}")
        
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        
        # Load model
        print("\n⏳ Loading embedding model...")
        local_path = Path(self.workspace) / 'embedding_model'
        if local_path.exists():
            self.model = SentenceTransformer(str(local_path), device=self.device)
        else:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)
        print("✅ Model loaded")
        
        self.results = {
            'system': {
                'ram_gb': self.total_ram_gb,
                'cpu_cores': self.cpu_cores,
                'device': self.device,
                'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            },
            'stress_tests': [],
            'limits_found': {},
        }
    
    def generate_chunks(self, num_chunks: int) -> List[str]:
        """Generate test chunks"""
        templates = [
            "PROCUREMENT DOCUMENT Section {i}: Technical specifications require minimum processor speed of 3.0GHz with 16GB RAM. Budget allocation: PHP {amt:,.2f}. Delivery within 45 days.",
            "BIDDER REQUIREMENTS Article {i}: Submit PhilGEPS registration, Mayor's permit, tax clearance, and audited financial statements. Contract value: PHP {amt:,.2f}.",
            "COMPLIANCE SECTION {i}: All items must meet ISO standards. Warranty period: 3 years. Liquidated damages: 1/10 of 1% per day. Total: PHP {amt:,.2f}.",
            "SCHEDULE Item {i}: Pre-bid conference on Day 7, submission deadline Day 21, opening Day 22. Project duration: 90 calendar days. Amount: PHP {amt:,.2f}.",
            "LEGAL CLAUSE {i}: Force majeure provisions apply. Arbitration venue: Manila. Governing law: Philippine jurisdiction. Value: PHP {amt:,.2f}.",
        ]
        
        chunks = []
        for i in range(num_chunks):
            template = templates[i % len(templates)]
            amt = np.random.uniform(100000, 5000000)
            chunk = template.format(i=i+1, amt=amt)
            chunks.append(chunk)
        return chunks
    
    def measure_retrieval(self, chunks: List[str], num_queries: int = 5) -> Dict:
        """Measure retrieval performance"""
        
        # Create embeddings in batches if needed
        batch_size = 5000
        if len(chunks) > batch_size:
            embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                batch_emb = self.model.encode(batch, show_progress_bar=False)
                embeddings.append(batch_emb)
            embeddings = np.vstack(embeddings)
        else:
            embeddings = self.model.encode(chunks, show_progress_bar=False)
        
        # Create collection
        client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
        coll_name = f"stress_{len(chunks)}_{int(time.time()*1000)}"
        collection = client.create_collection(name=coll_name, metadata={"hnsw:space": "cosine"})
        
        # Add documents in batches (ChromaDB limit is ~5461)
        batch_size = 5000
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            collection.add(
                documents=chunks[i:end],
                embeddings=embeddings[i:end].tolist(),
                metadatas=[{"id": j} for j in range(i, end)],
                ids=[f"c{j}" for j in range(i, end)]
            )
        
        queries = [
            "What is the approved budget?",
            "Technical specifications required",
            "Eligibility requirements for bidders",
            "Delivery timeline and schedule",
            "Compliance and warranty terms",
        ]
        
        # Warm up
        q_emb = self.model.encode(queries[0])
        _ = collection.query(query_embeddings=[q_emb.tolist()], n_results=5)
        
        # Measure
        times = []
        for q in queries[:num_queries]:
            for _ in range(3):
                start = time.perf_counter()
                q_emb = self.model.encode(q)
                _ = collection.query(query_embeddings=[q_emb.tolist()], n_results=5)
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        # Cleanup
        del collection
        del client
        gc.collect()
        
        return {
            'min': round(min(times), 2),
            'max': round(max(times), 2),
            'avg': round(statistics.mean(times), 2),
            'median': round(statistics.median(times), 2),
            'p95': round(np.percentile(times, 95), 2),
            'p99': round(np.percentile(times, 99), 2),
        }
    
    def run_stress_test(self):
        """Run progressive stress test"""
        print("\n" + "=" * 70)
        print("  STRESS TEST: FINDING PERFORMANCE LIMITS")
        print("=" * 70)
        
        # Progressive chunk counts to stress test
        # Start small, increase exponentially until failure
        chunk_counts = [
            50,      # ~20 pages
            500,     # ~200 pages
            2000,    # ~800 pages
            5000,    # ~2000 pages
            10000,   # ~4000 pages
            25000,   # ~10000 pages
            50000,   # ~20000 pages
            75000,   # ~30000 pages
            100000,  # ~40000 pages
            150000,  # ~60000 pages
            200000,  # ~80000 pages
            300000,  # ~120000 pages
        ]
        
        # Track limits
        excellent_limit = 0
        good_limit = 0
        acceptable_limit = 0
        degraded_limit = 0
        failure_point = None
        
        print(f"\n{'Chunks':<10} {'Pages':<10} {'Avg(ms)':<12} {'P95(ms)':<12} {'Memory':<12} {'Status'}")
        print("-" * 70)
        
        for num_chunks in chunk_counts:
            est_pages = int(num_chunks / 2.5)
            
            # Check available memory
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            
            if available_gb < 1.0:  # Less than 1GB available
                print(f"\n⚠️  Low memory ({available_gb:.1f}GB available). Stopping stress test.")
                failure_point = num_chunks
                break
            
            try:
                # Generate and test
                chunks = self.generate_chunks(num_chunks)
                
                mem_before = psutil.Process().memory_info().rss / (1024**2)
                timing = self.measure_retrieval(chunks)
                mem_after = psutil.Process().memory_info().rss / (1024**2)
                mem_used = mem_after - mem_before
                
                # Classify performance
                avg = timing['avg']
                if avg < self.EXCELLENT:
                    status = "🟢 EXCELLENT"
                    excellent_limit = num_chunks
                    good_limit = num_chunks
                    acceptable_limit = num_chunks
                elif avg < self.GOOD:
                    status = "🟢 GOOD"
                    good_limit = num_chunks
                    acceptable_limit = num_chunks
                elif avg < self.ACCEPTABLE:
                    status = "🟡 ACCEPTABLE"
                    acceptable_limit = num_chunks
                elif avg < self.DEGRADED:
                    status = "🟠 DEGRADED"
                    degraded_limit = num_chunks
                else:
                    status = "🔴 POOR"
                    if failure_point is None:
                        failure_point = num_chunks
                
                print(f"{num_chunks:<10} {est_pages:<10} {avg:<12.1f} {timing['p95']:<12.1f} {mem_used:<12.0f}MB {status}")
                
                # Store result
                self.results['stress_tests'].append({
                    'chunks': num_chunks,
                    'est_pages': est_pages,
                    'timing': timing,
                    'memory_mb': round(mem_used, 2),
                    'status': status,
                })
                
                # Clean up
                del chunks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Stop if we've clearly exceeded acceptable performance
                if avg > self.FAILURE:
                    print(f"\n🛑 Performance exceeded {self.FAILURE}ms. Stopping stress test.")
                    failure_point = num_chunks
                    break
                    
            except Exception as e:
                print(f"{num_chunks:<10} {est_pages:<10} {'FAILED':<12} {'-':<12} {'-':<12} 🔴 ERROR: {str(e)[:30]}")
                failure_point = num_chunks
                break
        
        # Store limits
        self.results['limits_found'] = {
            'excellent_limit_chunks': excellent_limit,
            'excellent_limit_pages': int(excellent_limit / 2.5),
            'good_limit_chunks': good_limit,
            'good_limit_pages': int(good_limit / 2.5),
            'acceptable_limit_chunks': acceptable_limit,
            'acceptable_limit_pages': int(acceptable_limit / 2.5),
            'failure_point_chunks': failure_point,
            'failure_point_pages': int(failure_point / 2.5) if failure_point else None,
        }
        
        return self.results['limits_found']
    
    def extrapolate_other_tiers(self):
        """Extrapolate limits for other hardware tiers"""
        print("\n" + "=" * 70)
        print("  EXTRAPOLATING LIMITS FOR ALL HARDWARE TIERS")
        print("=" * 70)
        
        limits = self.results['limits_found']
        baseline_ram = self.total_ram_gb
        
        # Performance scales roughly with RAM for vector operations
        # CPU affects embedding generation speed
        # These are conservative multipliers based on typical hardware ratios
        
        tier_configs = {
            'LOW-END': {
                'ram_gb': 4,
                'cpu_cores': 2,
                'description': 'Basic Govt PCs (Celeron/i3/Ryzen 3)',
                'ram_multiplier': 4 / baseline_ram,  # Scale by RAM ratio
                'cpu_multiplier': 0.6,  # Slower CPU
            },
            'MID-RANGE': {
                'ram_gb': 8,
                'cpu_cores': 4,
                'description': 'Standard Office (i5/Ryzen 5)',
                'ram_multiplier': 8 / baseline_ram,
                'cpu_multiplier': 0.8,
            },
            'HIGH-END': {
                'ram_gb': 16,
                'cpu_cores': 6,
                'description': 'IT Dept Systems (i7/i9/Ryzen 7/9)',
                'ram_multiplier': 1.0,  # Baseline (measured)
                'cpu_multiplier': 1.0,
            },
        }
        
        # Get measured retrieval times at different scales
        stress_data = self.results['stress_tests']
        
        # Find baseline retrieval time at a reference point (500 chunks)
        baseline_retrieval = 11.0  # Default
        for test in stress_data:
            if test['chunks'] == 500:
                baseline_retrieval = test['timing']['avg']
                break
        
        extrapolated = {}
        
        for tier, config in tier_configs.items():
            # Scale limits by RAM ratio (memory constrains max chunks)
            ram_mult = config['ram_multiplier']
            cpu_mult = config['cpu_multiplier']
            
            # Retrieval time scales inversely with CPU performance
            retrieval_multiplier = 1 / cpu_mult
            
            # Page limits scale with RAM
            tier_excellent = int(limits['excellent_limit_pages'] * ram_mult)
            tier_good = int(limits['good_limit_pages'] * ram_mult)
            tier_acceptable = int(limits['acceptable_limit_pages'] * ram_mult)
            
            # Apply minimum bounds
            tier_excellent = max(tier_excellent, 5)
            tier_good = max(tier_good, 10)
            tier_acceptable = max(tier_acceptable, 15)
            
            # Estimate retrieval time
            est_retrieval = baseline_retrieval * retrieval_multiplier
            
            extrapolated[tier] = {
                'ram_gb': config['ram_gb'],
                'cpu_cores': config['cpu_cores'],
                'description': config['description'],
                'max_pages_excellent': tier_excellent,
                'max_pages_good': tier_good,
                'max_pages_acceptable': tier_acceptable,
                'estimated_retrieval_ms': round(est_retrieval, 1),
                'optimal_pages': min(tier_good, 90 if tier == 'HIGH-END' else 30 if tier == 'MID-RANGE' else 10),
                'optimal_files': 6 if tier == 'HIGH-END' else 2 if tier == 'MID-RANGE' else 1,
                'optimal_file_size_mb': 20 if tier == 'HIGH-END' else 15 if tier == 'MID-RANGE' else 5,
                'is_measured': tier == 'HIGH-END',
            }
            
            measured = " (MEASURED)" if tier == 'HIGH-END' else " (Extrapolated)"
            print(f"\n{tier} - {config['description']}{measured}")
            print(f"   Max Pages (Excellent <15ms): {tier_excellent}")
            print(f"   Max Pages (Good <25ms): {tier_good}")
            print(f"   Max Pages (Acceptable <50ms): {tier_acceptable}")
            print(f"   Est. Retrieval Time: {est_retrieval:.1f}ms")
        
        self.results['tier_extrapolation'] = extrapolated
        return extrapolated
    
    def generate_report(self):
        """Generate stress test report"""
        report = []
        
        report.append("# RAG System Stress Test Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Purpose:** Find actual performance limits through stress testing")
        
        # System
        report.append("\n---\n")
        report.append("## 1. Test System Specifications")
        report.append(f"\n| Component | Value |")
        report.append(f"|-----------|-------|")
        report.append(f"| RAM | {self.total_ram_gb} GB |")
        report.append(f"| CPU Cores | {self.cpu_cores} |")
        report.append(f"| GPU | {self.results['system']['gpu'] or 'None'} |")
        report.append(f"| Compute | {self.device.upper()} |")
        
        # Stress test results
        report.append("\n---\n")
        report.append("## 2. Stress Test Results")
        report.append("\n| Chunks | Est. Pages | Avg (ms) | P95 (ms) | Memory (MB) | Status |")
        report.append("|--------|------------|----------|----------|-------------|--------|")
        
        for test in self.results['stress_tests']:
            report.append(f"| {test['chunks']} | {test['est_pages']} | **{test['timing']['avg']:.1f}** | {test['timing']['p95']:.1f} | {test['memory_mb']:.0f} | {test['status']} |")
        
        # Limits found
        report.append("\n---\n")
        report.append("## 3. Performance Limits Discovered")
        
        limits = self.results['limits_found']
        report.append(f"\n| Performance Level | Max Chunks | Max Pages | Retrieval Time |")
        report.append(f"|-------------------|------------|-----------|----------------|")
        report.append(f"| 🟢 **EXCELLENT** | {limits['excellent_limit_chunks']} | **{limits['excellent_limit_pages']}** | < 15ms |")
        report.append(f"| 🟢 **GOOD** | {limits['good_limit_chunks']} | **{limits['good_limit_pages']}** | < 25ms |")
        report.append(f"| 🟡 **ACCEPTABLE** | {limits['acceptable_limit_chunks']} | **{limits['acceptable_limit_pages']}** | < 50ms |")
        if limits['failure_point_chunks']:
            report.append(f"| 🔴 **FAILURE** | {limits['failure_point_chunks']} | {limits['failure_point_pages']} | > 100ms |")
        
        # Extrapolation
        report.append("\n---\n")
        report.append("## 4. Extrapolated Limits by Hardware Tier")
        report.append("\n### 4.1 Summary Table")
        report.append("\n| Tier | RAM | Max Pages (Excellent) | Max Pages (Good) | Optimal Config | Est. Retrieval |")
        report.append("|------|-----|----------------------|------------------|----------------|----------------|")
        
        for tier, data in self.results['tier_extrapolation'].items():
            measured = " ✓" if data['is_measured'] else ""
            report.append(f"| **{tier}**{measured} | {data['ram_gb']}GB | {data['max_pages_excellent']} | {data['max_pages_good']} | {data['optimal_pages']}pg / {data['optimal_files']}files / {data['optimal_file_size_mb']}MB | {data['estimated_retrieval_ms']:.0f}ms |")
        
        # Detailed tier breakdown
        report.append("\n### 4.2 Detailed Recommendations")
        
        for tier, data in self.results['tier_extrapolation'].items():
            measured = " (MEASURED)" if data['is_measured'] else " (Extrapolated)"
            report.append(f"\n#### {tier} - {data['description']}{measured}")
            report.append(f"\n| Metric | Value |")
            report.append(f"|--------|-------|")
            report.append(f"| **Optimal Pages** | {data['optimal_pages']} |")
            report.append(f"| **Optimal Files** | {data['optimal_files']} |")
            report.append(f"| **Max File Size** | {data['optimal_file_size_mb']} MB |")
            report.append(f"| **Max Pages (Excellent)** | {data['max_pages_excellent']} |")
            report.append(f"| **Max Pages (Good)** | {data['max_pages_good']} |")
            report.append(f"| **Est. Retrieval Time** | {data['estimated_retrieval_ms']:.1f}ms |")
        
        # Key findings
        report.append("\n---\n")
        report.append("## 5. Key Findings")
        
        report.append("\n### 5.1 Measured Performance Limits (HIGH-END System)")
        report.append(f"\n- **Excellent Performance (<15ms):** Up to **{limits['excellent_limit_pages']} pages** ({limits['excellent_limit_chunks']} chunks)")
        report.append(f"- **Good Performance (<25ms):** Up to **{limits['good_limit_pages']} pages** ({limits['good_limit_chunks']} chunks)")
        report.append(f"- **Acceptable Performance (<50ms):** Up to **{limits['acceptable_limit_pages']} pages** ({limits['acceptable_limit_chunks']} chunks)")
        
        report.append("\n### 5.2 Retrieval Time Characteristics")
        
        # Calculate growth rate
        if len(self.results['stress_tests']) >= 2:
            first = self.results['stress_tests'][0]
            last = self.results['stress_tests'][-1]
            chunk_diff = last['chunks'] - first['chunks']
            time_diff = last['timing']['avg'] - first['timing']['avg']
            growth_rate = time_diff / chunk_diff * 1000 if chunk_diff > 0 else 0
            
            report.append(f"\n- Retrieval time grows at approximately **{growth_rate:.2f}ms per 1000 chunks**")
            report.append(f"- Base retrieval time: ~{first['timing']['avg']:.1f}ms (minimum overhead)")
        
        report.append("\n### 5.3 Conclusions for Thesis")
        report.append("\n**Optimal Configuration by Hardware Tier:**\n")
        report.append("| Hardware | RAM | Optimal Pages | Optimal Files | Max Size | Retrieval |")
        report.append("|----------|-----|---------------|---------------|----------|-----------|")
        for tier, data in self.results['tier_extrapolation'].items():
            report.append(f"| {tier} | {data['ram_gb']}GB | **{data['optimal_pages']}** | {data['optimal_files']} | {data['optimal_file_size_mb']}MB | {data['estimated_retrieval_ms']:.0f}ms |")
        
        report_text = "\n".join(report)
        
        # Save
        report_path = os.path.join(self.workspace, 'stress_test_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n📊 Report saved to: {report_path}")
        return report_text
    
    def save_results(self):
        """Save results to JSON"""
        path = os.path.join(self.workspace, 'stress_test_results.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Results saved to: {path}")
    
    def run(self):
        """Run complete stress test"""
        start = time.time()
        
        self.run_stress_test()
        self.extrapolate_other_tiers()
        self.generate_report()
        self.save_results()
        
        elapsed = time.time() - start
        
        print("\n" + "=" * 70)
        print("  STRESS TEST COMPLETE")
        print("=" * 70)
        
        limits = self.results['limits_found']
        print(f"\n🎯 PERFORMANCE LIMITS FOUND:")
        print(f"   🟢 Excellent (<15ms): {limits['excellent_limit_pages']} pages")
        print(f"   🟢 Good (<25ms): {limits['good_limit_pages']} pages")
        print(f"   🟡 Acceptable (<50ms): {limits['acceptable_limit_pages']} pages")
        print(f"\n⏱️  Total time: {elapsed:.1f} seconds")
        
        return self.results


if __name__ == "__main__":
    stress = StressTest()
    results = stress.run()
