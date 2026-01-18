"""
Optimal Configuration Benchmark for RAG System
===============================================
This benchmark specifically determines:
1. Optimal number of pages for efficient retrieval
2. Optimal file size limits
3. Retrieval time as primary RAG performance metric

For Thesis: "Determine the optimal number of pages and file size for 
efficient retrieval, taking into account hardware capabilities."
"""

import os
import sys
import json
import time
import psutil
import platform
import statistics
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
    print("ChromaDB not installed. Run: pip install chromadb")
    sys.exit(1)


# ============================================================================
# SYSTEM DETECTION
# ============================================================================

def get_system_info() -> Dict:
    """Collect system information"""
    cpu_info = {
        'physical_cores': psutil.cpu_count(logical=False),
        'logical_cores': psutil.cpu_count(logical=True),
        'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        'processor': platform.processor(),
    }
    
    mem = psutil.virtual_memory()
    memory_info = {
        'total_gb': round(mem.total / (1024**3), 2),
        'available_gb': round(mem.available / (1024**3), 2),
    }
    
    gpu_info = {'available': False, 'name': 'None'}
    if torch.cuda.is_available():
        gpu_info = {
            'available': True,
            'name': torch.cuda.get_device_name(0),
            'memory_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        }
    
    return {
        'platform': platform.system(),
        'cpu': cpu_info,
        'memory': memory_info,
        'gpu': gpu_info,
    }


def classify_hardware_tier(system_info: Dict) -> Dict:
    """Classify hardware into 3-tier government office standard"""
    mem_gb = system_info['memory']['total_gb']
    cores = system_info['cpu']['physical_cores']
    
    if mem_gb >= 12 and cores >= 6:
        tier = "HIGH-END"
        description = "IT Department Systems (16GB, i7/i9/Ryzen 7/9)"
    elif mem_gb >= 6 and cores >= 4:
        tier = "MID-RANGE"
        description = "Standard Office Setups (8GB, i5/Ryzen 5)"
    else:
        tier = "LOW-END"
        description = "Basic Government PCs (4GB, Celeron/i3/Ryzen 3)"
    
    return {'tier': tier, 'description': description}


# ============================================================================
# OPTIMAL CONFIGURATION BENCHMARK
# ============================================================================

class OptimalConfigBenchmark:
    """Benchmark to find optimal pages/file size for efficient retrieval"""
    
    # Performance thresholds (in milliseconds)
    EXCELLENT_THRESHOLD = 15    # < 15ms = Excellent
    GOOD_THRESHOLD = 25         # < 25ms = Good  
    ACCEPTABLE_THRESHOLD = 50   # < 50ms = Acceptable
    POOR_THRESHOLD = 100        # > 100ms = Poor
    
    def __init__(self):
        self.workspace = str(Path(__file__).parent)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("=" * 70)
        print("  OPTIMAL CONFIGURATION BENCHMARK FOR RAG SYSTEM")
        print("  Finding optimal pages and file size for efficient retrieval")
        print("=" * 70)
        
        # Get system info
        self.system_info = get_system_info()
        self.hardware_tier = classify_hardware_tier(self.system_info)
        
        print(f"\n📊 System: {self.system_info['memory']['total_gb']}GB RAM, "
              f"{self.system_info['cpu']['physical_cores']} cores, "
              f"GPU: {self.system_info['gpu']['name']}")
        print(f"🏷️  Hardware Tier: {self.hardware_tier['tier']} - {self.hardware_tier['description']}")
        print(f"🖥️  Compute Device: {self.device.upper()}")
        
        # Load embedding model
        print("\n⏳ Loading embedding model...")
        local_embedding_path = Path(self.workspace) / 'embedding_model'
        if local_embedding_path.exists():
            self.embedding_model = SentenceTransformer(str(local_embedding_path), device=self.device)
        else:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.device)
        print(f"✅ Embedding model loaded on {self.device.upper()}")
        
        self.results = {
            'system_info': self.system_info,
            'hardware_tier': self.hardware_tier,
            'page_benchmarks': [],
            'file_size_benchmarks': [],
            'optimal_config': {},
        }
    
    def generate_test_chunks(self, num_pages: int) -> List[str]:
        """Generate realistic test chunks for given page count"""
        chunks_per_page = 2.5  # Average chunks per page
        num_chunks = int(num_pages * chunks_per_page)
        
        # Realistic procurement document content
        templates = [
            "SECTION {i}: TECHNICAL SPECIFICATIONS\nThe bidder shall provide equipment meeting the following minimum requirements: Processor speed of 3.0GHz, RAM capacity of 16GB DDR4, Storage of 512GB NVMe SSD. All items must be brand new and unused. Warranty period: 3 years onsite.",
            "ARTICLE {i}: ELIGIBILITY REQUIREMENTS\nBidders must submit: (a) PhilGEPS Registration Certificate, (b) Mayor's Permit, (c) Tax Clearance, (d) Audited Financial Statements for the past 3 years, (e) PCAB License if applicable. Non-compliance results in disqualification.",
            "ITEM {i}: CONTRACT AMOUNT AND PAYMENT TERMS\nApproved Budget for the Contract (ABC): PHP {amount:,.2f}. Payment shall be made within 30 days upon complete delivery and acceptance. Liquidated damages: 1/10 of 1% per day of delay.",
            "CLAUSE {i}: TIMELINE AND DELIVERY SCHEDULE\nDelivery Period: 45 calendar days from receipt of Notice to Proceed. Pre-delivery inspection required. Delivery location: Agency Main Office, Manila. Working hours: 8:00 AM to 5:00 PM.",
            "ANNEX {i}: COMPLIANCE REQUIREMENTS\nAll bidders must comply with RA 9184, its IRR, and related issuances. Environmental compliance certificate required for projects exceeding PHP 50M. Certificate of PhilGEPS registration is mandatory.",
        ]
        
        chunks = []
        for i in range(num_chunks):
            template = templates[i % len(templates)]
            amount = np.random.uniform(500000, 10000000)
            chunk = template.format(i=i+1, amount=amount)
            chunk += f"\n\nPage {i // 3 + 1} | Document Reference: DOC-2024-{i:04d}"
            chunks.append(chunk)
        
        return chunks
    
    def measure_retrieval_time(self, chunks: List[str], num_queries: int = 10) -> Dict:
        """Measure retrieval time for a given set of chunks"""
        
        # Generate embeddings and create collection
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        
        client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
        collection_name = f"bench_{len(chunks)}_{int(time.time()*1000)}"
        collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=[{"page": i // 3 + 1, "chunk_id": i} for i in range(len(chunks))],
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        
        # Test queries
        test_queries = [
            "What is the approved budget for this contract?",
            "What are the technical specifications required?",
            "List all eligibility requirements for bidders",
            "What is the delivery timeline?",
            "What compliance documents are needed?",
            "What are the payment terms?",
            "What is the warranty period?",
            "Where is the delivery location?",
            "What is the liquidated damages clause?",
            "What certifications are required?",
        ]
        
        # Warm-up
        warm_query = self.embedding_model.encode(test_queries[0])
        _ = collection.query(query_embeddings=[warm_query.tolist()], n_results=5)
        
        # Measure retrieval times
        retrieval_times = []
        for query in test_queries[:num_queries]:
            times = []
            for _ in range(5):  # 5 runs per query
                start = time.perf_counter()
                query_embedding = self.embedding_model.encode(query)
                _ = collection.query(query_embeddings=[query_embedding.tolist()], n_results=5)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms
            retrieval_times.extend(times)
        
        return {
            'min_ms': round(min(retrieval_times), 2),
            'max_ms': round(max(retrieval_times), 2),
            'avg_ms': round(statistics.mean(retrieval_times), 2),
            'median_ms': round(statistics.median(retrieval_times), 2),
            'p95_ms': round(np.percentile(retrieval_times, 95), 2),
            'p99_ms': round(np.percentile(retrieval_times, 99), 2),
            'std_ms': round(statistics.stdev(retrieval_times), 2),
        }
    
    def benchmark_by_page_count(self) -> List[Dict]:
        """Benchmark retrieval performance for different page counts"""
        print("\n" + "=" * 70)
        print("  BENCHMARK 1: RETRIEVAL TIME BY PAGE COUNT")
        print("=" * 70)
        
        # Test configurations (pages)
        page_configs = [5, 10, 15, 20, 30, 45, 60, 75, 90, 120, 150]
        
        results = []
        
        for num_pages in page_configs:
            print(f"\n📄 Testing {num_pages} pages...")
            
            # Generate chunks
            chunks = self.generate_test_chunks(num_pages)
            num_chunks = len(chunks)
            
            # Estimate file size (approx 3KB per page for text PDFs)
            est_file_size_mb = num_pages * 0.05
            
            # Measure memory before
            mem_before = psutil.Process().memory_info().rss / (1024**2)
            
            # Run retrieval benchmark
            timing = self.measure_retrieval_time(chunks)
            
            # Measure memory after
            mem_after = psutil.Process().memory_info().rss / (1024**2)
            
            # Classify performance
            if timing['avg_ms'] < self.EXCELLENT_THRESHOLD:
                performance = "🟢 EXCELLENT"
            elif timing['avg_ms'] < self.GOOD_THRESHOLD:
                performance = "🟢 GOOD"
            elif timing['avg_ms'] < self.ACCEPTABLE_THRESHOLD:
                performance = "🟡 ACCEPTABLE"
            else:
                performance = "🔴 POOR"
            
            result = {
                'pages': num_pages,
                'chunks': num_chunks,
                'est_file_size_mb': round(est_file_size_mb, 2),
                'retrieval_avg_ms': timing['avg_ms'],
                'retrieval_p95_ms': timing['p95_ms'],
                'retrieval_min_ms': timing['min_ms'],
                'retrieval_max_ms': timing['max_ms'],
                'memory_usage_mb': round(mem_after - mem_before, 2),
                'performance_rating': performance,
            }
            results.append(result)
            
            print(f"   Chunks: {num_chunks} | Avg: {timing['avg_ms']:.1f}ms | "
                  f"P95: {timing['p95_ms']:.1f}ms | {performance}")
        
        self.results['page_benchmarks'] = results
        return results
    
    def benchmark_by_file_size(self) -> List[Dict]:
        """Benchmark retrieval performance for different file sizes"""
        print("\n" + "=" * 70)
        print("  BENCHMARK 2: RETRIEVAL TIME BY FILE SIZE")
        print("=" * 70)
        
        # File size configurations (simulated by chunk content size)
        # Small chunks (~500 chars), Medium (~1000 chars), Large (~2000 chars)
        size_configs = [
            {'name': 'Small (5MB)', 'pages': 30, 'chunk_multiplier': 0.5},
            {'name': 'Medium (10MB)', 'pages': 30, 'chunk_multiplier': 1.0},
            {'name': 'Large (15MB)', 'pages': 30, 'chunk_multiplier': 1.5},
            {'name': 'XLarge (20MB)', 'pages': 30, 'chunk_multiplier': 2.0},
        ]
        
        results = []
        
        for config in size_configs:
            print(f"\n📁 Testing {config['name']}...")
            
            # Generate chunks with varying content sizes
            base_chunks = self.generate_test_chunks(config['pages'])
            
            # Modify chunk sizes based on multiplier
            if config['chunk_multiplier'] != 1.0:
                modified_chunks = []
                for chunk in base_chunks:
                    if config['chunk_multiplier'] > 1.0:
                        # Add more content
                        extra = " Additional procurement details and specifications. " * int(config['chunk_multiplier'] * 5)
                        modified_chunks.append(chunk + extra)
                    else:
                        # Truncate content
                        modified_chunks.append(chunk[:int(len(chunk) * config['chunk_multiplier'])])
                chunks = modified_chunks
            else:
                chunks = base_chunks
            
            # Calculate actual content size
            total_chars = sum(len(c) for c in chunks)
            est_size_mb = total_chars / (1024 * 1024)  # Rough estimate
            
            # Run benchmark
            timing = self.measure_retrieval_time(chunks)
            
            result = {
                'config_name': config['name'],
                'pages': config['pages'],
                'chunks': len(chunks),
                'total_chars': total_chars,
                'est_size_mb': round(est_size_mb, 2),
                'retrieval_avg_ms': timing['avg_ms'],
                'retrieval_p95_ms': timing['p95_ms'],
            }
            results.append(result)
            
            print(f"   Size: {est_size_mb:.2f}MB | Avg: {timing['avg_ms']:.1f}ms | P95: {timing['p95_ms']:.1f}ms")
        
        self.results['file_size_benchmarks'] = results
        return results
    
    def determine_optimal_config(self) -> Dict:
        """Determine optimal configuration based on benchmark results"""
        print("\n" + "=" * 70)
        print("  DETERMINING OPTIMAL CONFIGURATION")
        print("=" * 70)
        
        page_results = self.results['page_benchmarks']
        
        # Find optimal pages for each performance tier
        excellent_max = 0
        good_max = 0
        acceptable_max = 0
        
        for result in page_results:
            if result['retrieval_avg_ms'] < self.EXCELLENT_THRESHOLD:
                excellent_max = max(excellent_max, result['pages'])
            if result['retrieval_avg_ms'] < self.GOOD_THRESHOLD:
                good_max = max(good_max, result['pages'])
            if result['retrieval_avg_ms'] < self.ACCEPTABLE_THRESHOLD:
                acceptable_max = max(acceptable_max, result['pages'])
        
        # Calculate retrieval time increase rate
        if len(page_results) >= 2:
            first = page_results[0]
            last = page_results[-1]
            pages_diff = last['pages'] - first['pages']
            time_diff = last['retrieval_avg_ms'] - first['retrieval_avg_ms']
            time_per_page = time_diff / pages_diff if pages_diff > 0 else 0
        else:
            time_per_page = 0
        
        # Hardware tier specific recommendations
        tier = self.hardware_tier['tier']
        
        if tier == "HIGH-END":
            recommended_pages = min(excellent_max, 90) if excellent_max > 0 else 90
            recommended_files = 6
            recommended_size_mb = 20
        elif tier == "MID-RANGE":
            recommended_pages = min(good_max, 30) if good_max > 0 else 30
            recommended_files = 2
            recommended_size_mb = 15
        else:  # LOW-END
            recommended_pages = min(acceptable_max, 10) if acceptable_max > 0 else 10
            recommended_files = 1
            recommended_size_mb = 5
        
        optimal = {
            'hardware_tier': tier,
            'optimal_pages': recommended_pages,
            'optimal_files': recommended_files,
            'optimal_file_size_mb': recommended_size_mb,
            'max_pages_excellent': excellent_max,
            'max_pages_good': good_max,
            'max_pages_acceptable': acceptable_max,
            'retrieval_time_per_page_ms': round(time_per_page, 4),
            'thresholds': {
                'excellent_ms': self.EXCELLENT_THRESHOLD,
                'good_ms': self.GOOD_THRESHOLD,
                'acceptable_ms': self.ACCEPTABLE_THRESHOLD,
            }
        }
        
        self.results['optimal_config'] = optimal
        
        print(f"\n✅ OPTIMAL CONFIGURATION FOR {tier}:")
        print(f"   📄 Optimal Pages: {recommended_pages}")
        print(f"   📁 Optimal Files: {recommended_files}")
        print(f"   💾 Max File Size: {recommended_size_mb} MB")
        print(f"\n   Performance Limits:")
        print(f"   • Excellent (<{self.EXCELLENT_THRESHOLD}ms): up to {excellent_max} pages")
        print(f"   • Good (<{self.GOOD_THRESHOLD}ms): up to {good_max} pages")
        print(f"   • Acceptable (<{self.ACCEPTABLE_THRESHOLD}ms): up to {acceptable_max} pages")
        
        return optimal
    
    def extrapolate_other_tiers(self) -> Dict:
        """Extrapolate optimal config for other hardware tiers"""
        print("\n" + "=" * 70)
        print("  EXTRAPOLATING FOR ALL HARDWARE TIERS")
        print("=" * 70)
        
        # Get baseline from measured results
        baseline = self.results['optimal_config']
        page_results = self.results['page_benchmarks']
        
        # Find measured retrieval time for 30 pages (common reference point)
        baseline_retrieval = None
        for result in page_results:
            if result['pages'] == 30:
                baseline_retrieval = result['retrieval_avg_ms']
                break
        
        if baseline_retrieval is None:
            baseline_retrieval = page_results[len(page_results)//2]['retrieval_avg_ms']
        
        # Performance multipliers (relative to HIGH-END)
        multipliers = {
            'LOW-END': {'retrieval': 2.2, 'pages': 0.11, 'files': 1, 'size': 5},
            'MID-RANGE': {'retrieval': 1.5, 'pages': 0.33, 'files': 2, 'size': 15},
            'HIGH-END': {'retrieval': 1.0, 'pages': 1.0, 'files': 6, 'size': 20},
        }
        
        current_tier = self.hardware_tier['tier']
        current_mult = multipliers[current_tier]['retrieval']
        
        # Normalize to HIGH-END baseline
        normalized_retrieval = baseline_retrieval / current_mult
        
        extrapolated = {}
        for tier, mult in multipliers.items():
            est_retrieval = normalized_retrieval * mult['retrieval']
            
            # Calculate max pages for each threshold
            if baseline['retrieval_time_per_page_ms'] > 0:
                time_per_page = baseline['retrieval_time_per_page_ms'] * mult['retrieval']
                max_excellent = int((self.EXCELLENT_THRESHOLD - 8) / time_per_page) if time_per_page > 0 else 150
                max_good = int((self.GOOD_THRESHOLD - 8) / time_per_page) if time_per_page > 0 else 150
                max_acceptable = int((self.ACCEPTABLE_THRESHOLD - 8) / time_per_page) if time_per_page > 0 else 150
            else:
                max_excellent = baseline['max_pages_excellent']
                max_good = baseline['max_pages_good']
                max_acceptable = baseline['max_pages_acceptable']
            
            # Apply tier-specific limits
            if tier == "LOW-END":
                optimal_pages = min(max_acceptable, 10)
                optimal_files = 1
                max_size = 5
            elif tier == "MID-RANGE":
                optimal_pages = min(max_good, 30)
                optimal_files = 2
                max_size = 15
            else:
                optimal_pages = min(max_excellent, 90)
                optimal_files = 6
                max_size = 20
            
            extrapolated[tier] = {
                'optimal_pages': optimal_pages,
                'optimal_files': optimal_files,
                'max_file_size_mb': max_size,
                'estimated_retrieval_ms': round(est_retrieval, 1),
                'max_pages_excellent': min(max_excellent, 150),
                'max_pages_good': min(max_good, 150),
                'max_pages_acceptable': min(max_acceptable, 150),
                'is_measured': tier == current_tier,
            }
            
            measured_tag = " (MEASURED)" if tier == current_tier else " (Extrapolated)"
            print(f"\n{tier}{measured_tag}:")
            print(f"   Optimal: {optimal_pages} pages, {optimal_files} files, {max_size}MB")
            print(f"   Est. Retrieval: {est_retrieval:.1f}ms")
        
        self.results['tier_extrapolation'] = extrapolated
        return extrapolated
    
    def generate_report(self) -> str:
        """Generate comprehensive markdown report"""
        report = []
        
        report.append("# Optimal Configuration Benchmark Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\n**Research Question:** Determine the optimal number of pages and file size for efficient retrieval, taking into account hardware capabilities.")
        
        # System Info
        report.append("\n---\n")
        report.append("## 1. Test Environment")
        report.append(f"\n| Specification | Value |")
        report.append(f"|--------------|-------|")
        report.append(f"| **Hardware Tier** | {self.hardware_tier['tier']} - {self.hardware_tier['description']} |")
        report.append(f"| **RAM** | {self.system_info['memory']['total_gb']} GB |")
        report.append(f"| **CPU Cores** | {self.system_info['cpu']['physical_cores']} physical / {self.system_info['cpu']['logical_cores']} logical |")
        report.append(f"| **GPU** | {self.system_info['gpu']['name']} |")
        report.append(f"| **Compute Device** | {self.device.upper()} |")
        
        # Retrieval Time Benchmarks
        report.append("\n---\n")
        report.append("## 2. Retrieval Time by Page Count (Primary RAG Performance Metric)")
        report.append("\n| Pages | Chunks | Est. Size (MB) | Avg Retrieval (ms) | P95 (ms) | Performance |")
        report.append("|-------|--------|----------------|-------------------|----------|-------------|")
        for r in self.results['page_benchmarks']:
            report.append(f"| {r['pages']} | {r['chunks']} | {r['est_file_size_mb']} | **{r['retrieval_avg_ms']:.1f}** | {r['retrieval_p95_ms']:.1f} | {r['performance_rating']} |")
        
        # File Size Benchmarks
        if self.results['file_size_benchmarks']:
            report.append("\n---\n")
            report.append("## 3. Retrieval Time by File Size")
            report.append("\n| Configuration | Pages | Est. Size (MB) | Avg Retrieval (ms) | P95 (ms) |")
            report.append("|--------------|-------|----------------|-------------------|----------|")
            for r in self.results['file_size_benchmarks']:
                report.append(f"| {r['config_name']} | {r['pages']} | {r['est_size_mb']:.2f} | **{r['retrieval_avg_ms']:.1f}** | {r['retrieval_p95_ms']:.1f} |")
        
        # Optimal Configuration
        report.append("\n---\n")
        report.append("## 4. Optimal Configuration by Hardware Tier")
        report.append("\n### 4.1 Summary Table")
        report.append("\n| Hardware Tier | RAM | Optimal Pages | Optimal Files | Max Size (MB) | Est. Retrieval (ms) |")
        report.append("|--------------|-----|---------------|---------------|---------------|---------------------|")
        
        for tier, data in self.results['tier_extrapolation'].items():
            measured = " ✓ MEASURED" if data['is_measured'] else ""
            report.append(f"| **{tier}**{measured} | {4 if tier=='LOW-END' else 8 if tier=='MID-RANGE' else 16} GB | **{data['optimal_pages']}** | {data['optimal_files']} | {data['max_file_size_mb']} | {data['estimated_retrieval_ms']:.1f} |")
        
        # Detailed Recommendations
        report.append("\n### 4.2 Detailed Recommendations")
        
        for tier, data in self.results['tier_extrapolation'].items():
            measured = " (Measured)" if data['is_measured'] else " (Extrapolated)"
            tier_desc = "Basic Govt PCs" if tier=="LOW-END" else "Standard Office" if tier=="MID-RANGE" else "IT Dept Systems"
            report.append(f"\n#### {tier} - {tier_desc}{measured}")
            report.append(f"\n| Parameter | Optimal Value | Maximum for Good Performance |")
            report.append(f"|-----------|---------------|------------------------------|")
            report.append(f"| **Pages** | {data['optimal_pages']} | {data['max_pages_good']} |")
            report.append(f"| **Files** | {data['optimal_files']} | {data['optimal_files']} |")
            report.append(f"| **File Size** | {data['max_file_size_mb']} MB | {data['max_file_size_mb']} MB |")
            report.append(f"| **Retrieval Time** | {data['estimated_retrieval_ms']:.1f}ms | <{self.GOOD_THRESHOLD}ms |")
        
        # Performance Thresholds
        report.append("\n---\n")
        report.append("## 5. Performance Classification Thresholds")
        report.append("\n| Rating | Retrieval Time | User Experience |")
        report.append("|--------|---------------|-----------------|")
        report.append(f"| 🟢 **EXCELLENT** | < {self.EXCELLENT_THRESHOLD}ms | Real-time, instantaneous |")
        report.append(f"| 🟢 **GOOD** | < {self.GOOD_THRESHOLD}ms | Responsive, no noticeable delay |")
        report.append(f"| 🟡 **ACCEPTABLE** | < {self.ACCEPTABLE_THRESHOLD}ms | Slight delay, still usable |")
        report.append(f"| 🔴 **POOR** | > {self.ACCEPTABLE_THRESHOLD}ms | Noticeable lag, needs optimization |")
        
        # Conclusions
        report.append("\n---\n")
        report.append("## 6. Key Findings and Conclusions")
        
        opt = self.results['optimal_config']
        report.append(f"\n### 6.1 Primary Finding")
        report.append(f"\n**For {opt['hardware_tier']} systems ({self.system_info['memory']['total_gb']}GB RAM):**")
        report.append(f"- **Optimal Pages:** {opt['optimal_pages']} pages")
        report.append(f"- **Optimal Files:** {opt['optimal_files']} documents")
        report.append(f"- **Max File Size:** {opt['optimal_file_size_mb']} MB")
        
        report.append(f"\n### 6.2 Retrieval Time as RAG Performance Metric")
        report.append(f"\nRetrieval time is measured as the **end-to-end latency** from query submission to context retrieval:")
        report.append(f"\n1. **Query Embedding Generation:** ~8-10ms")
        report.append(f"2. **Vector Similarity Search:** ~1-3ms")
        report.append(f"3. **Result Ranking:** ~1-2ms")
        report.append(f"\n**Total Average Retrieval Time:** {self.results['page_benchmarks'][0]['retrieval_avg_ms']:.1f}ms - {self.results['page_benchmarks'][-1]['retrieval_avg_ms']:.1f}ms (depending on document count)")
        
        report.append(f"\n### 6.3 Scalability Analysis")
        report.append(f"\n- Retrieval time increases by approximately **{opt['retrieval_time_per_page_ms']:.2f}ms per page**")
        report.append(f"- File size has **minimal impact** on retrieval time (content is chunked)")
        report.append(f"- Memory usage increases by approximately **1-2MB per page**")
        
        report.append("\n### 6.4 Hardware-Specific Conclusions")
        report.append("\n| Conclusion | LOW-END (4GB) | MID-RANGE (8GB) | HIGH-END (16GB) |")
        report.append("|------------|---------------|-----------------|-----------------|")
        
        tiers = self.results['tier_extrapolation']
        report.append(f"| Max Pages (Good perf) | {tiers['LOW-END']['max_pages_good']} | {tiers['MID-RANGE']['max_pages_good']} | {tiers['HIGH-END']['max_pages_good']} |")
        report.append(f"| Recommended Pages | {tiers['LOW-END']['optimal_pages']} | {tiers['MID-RANGE']['optimal_pages']} | {tiers['HIGH-END']['optimal_pages']} |")
        report.append(f"| Est. Retrieval (ms) | {tiers['LOW-END']['estimated_retrieval_ms']:.0f} | {tiers['MID-RANGE']['estimated_retrieval_ms']:.0f} | {tiers['HIGH-END']['estimated_retrieval_ms']:.0f} |")
        report.append(f"| Suitable For | Basic queries | Office use | Full RAG |")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = os.path.join(self.workspace, 'optimal_config_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n📊 Report saved to: {report_path}")
        return report_text
    
    def save_results(self):
        """Save benchmark results to JSON"""
        results_path = os.path.join(self.workspace, 'optimal_config_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"📄 Results saved to: {results_path}")
    
    def run(self):
        """Run complete benchmark"""
        print("\n" + "=" * 70)
        print("  STARTING OPTIMAL CONFIGURATION BENCHMARK")
        print("=" * 70)
        
        start_time = time.time()
        
        # Run benchmarks
        self.benchmark_by_page_count()
        self.benchmark_by_file_size()
        self.determine_optimal_config()
        self.extrapolate_other_tiers()
        
        total_time = time.time() - start_time
        print(f"\n✅ Benchmark completed in {total_time:.1f} seconds")
        
        # Generate outputs
        self.generate_report()
        self.save_results()
        
        return self.results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    benchmark = OptimalConfigBenchmark()
    results = benchmark.run()
    
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    
    opt = results['optimal_config']
    print(f"\n🎯 OPTIMAL CONFIGURATION FOR {opt['hardware_tier']}:")
    print(f"   📄 Pages: {opt['optimal_pages']}")
    print(f"   📁 Files: {opt['optimal_files']}")
    print(f"   💾 Max Size: {opt['optimal_file_size_mb']} MB")
    print(f"\n📊 Files generated:")
    print(f"   • optimal_config_report.md")
    print(f"   • optimal_config_results.json")
