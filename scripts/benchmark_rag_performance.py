"""
================================================================================
RAG SYSTEM PERFORMANCE BENCHMARK SCRIPT
================================================================================
Tests the RAG system's retrieval and inference performance using ground truth
test data from the test folder.

Metrics Measured:
    - Document Processing Time
    - Embedding Generation Time
    - Retrieval Time (vector search)
    - Inference Time (LLM generation)
    - Total Response Time
    - Answer Accuracy (compared to ground truth)

Usage:
    python scripts/benchmark_rag_performance.py
================================================================================
"""

import os
import sys
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import statistics

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_backend import RAGSession, SessionManager, PDFProcessor, CONFIG


class RAGBenchmark:
    """
    Benchmark suite for testing RAG system performance.
    """
    
    def __init__(self, test_dir: str = "./test"):
        """
        Initialize benchmark with test directory.
        
        Args:
            test_dir: Path to directory containing test PDFs and ground truth.
        """
        self.test_dir = Path(test_dir)
        self.ground_truth_file = self.test_dir / "test1-ground-truth.json"
        self.results = {
            'system_info': {},
            'document_processing': [],
            'query_performance': [],
            'accuracy_results': [],
            'summary': {}
        }
        
        # Load ground truth
        self.ground_truth = self._load_ground_truth()
        
        # Initialize session manager
        self.session_manager = SessionManager(sessions_dir="./benchmark_sessions")
        
    def _load_ground_truth(self) -> Dict:
        """Load ground truth Q&A pairs from JSON file."""
        if self.ground_truth_file.exists():
            with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"⚠️ Ground truth file not found: {self.ground_truth_file}")
            return {}
    
    def _get_system_info(self) -> Dict:
        """Collect system information for the benchmark report."""
        import platform
        import psutil
        
        gpu_info = "N/A"
        gpu_memory = "N/A"
        if torch.cuda.is_available():
            gpu_info = torch.cuda.get_device_name(0)
            gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        
        return {
            'platform': platform.system(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'gpu_name': gpu_info,
            'gpu_memory': gpu_memory,
            'ram_total': f"{psutil.virtual_memory().total / 1e9:.2f} GB",
            'ram_available': f"{psutil.virtual_memory().available / 1e9:.2f} GB",
            'cpu_count': psutil.cpu_count(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_accuracy(self, response: str, expected: str) -> float:
        """
        Calculate accuracy score between response and expected answer.
        
        Uses keyword matching to determine if the key information is present.
        
        Args:
            response: Generated response from RAG system.
            expected: Expected answer from ground truth.
            
        Returns:
            Accuracy score between 0.0 and 1.0
        """
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Extract key terms from expected answer (numbers, dates, amounts)
        import re
        
        # Find monetary values
        money_pattern = r'(?:php|₱)?\s*[\d,]+(?:\.\d{2})?'
        expected_money = re.findall(money_pattern, expected_lower)
        
        # Find dates
        date_pattern = r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s*\d{4}'
        expected_dates = re.findall(date_pattern, expected_lower)
        
        # Find times
        time_pattern = r'\d{1,2}:\d{2}\s*(?:am|pm|a\.m\.|p\.m\.)?'
        expected_times = re.findall(time_pattern, expected_lower)
        
        # Find numbers
        number_pattern = r'\d+(?:,\d{3})*(?:\.\d+)?'
        expected_numbers = re.findall(number_pattern, expected_lower)
        
        # Calculate matches
        total_terms = 0
        matched_terms = 0
        
        # Check money matches
        for money in expected_money:
            total_terms += 1
            # Clean the money value for comparison
            clean_money = re.sub(r'[^\d.]', '', money)
            if clean_money and clean_money in re.sub(r'[^\d.]', '', response_lower):
                matched_terms += 1
        
        # Check date matches
        for date in expected_dates:
            total_terms += 1
            # Extract month and day
            date_parts = date.split()
            if len(date_parts) >= 2:
                if date_parts[0] in response_lower and date_parts[1].rstrip(',') in response_lower:
                    matched_terms += 1
        
        # Check time matches
        for t in expected_times:
            total_terms += 1
            clean_time = t.replace(' ', '').replace('.', '').lower()
            if clean_time in response_lower.replace(' ', '').replace('.', ''):
                matched_terms += 1
        
        # If no specific terms found, use simple keyword overlap
        if total_terms == 0:
            expected_words = set(expected_lower.split())
            response_words = set(response_lower.split())
            # Remove common words
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'for', 'of', 'to', 'and', 'in', 'on', 'at'}
            expected_words -= stopwords
            response_words -= stopwords
            
            if expected_words:
                overlap = expected_words & response_words
                return len(overlap) / len(expected_words)
            return 0.0
        
        return matched_terms / total_terms if total_terms > 0 else 0.0
    
    def benchmark_document_processing(self, session: RAGSession) -> List[Dict]:
        """
        Benchmark document processing (upload and indexing).
        
        Args:
            session: RAGSession instance to use for testing.
            
        Returns:
            List of processing results for each document.
        """
        print("\n" + "=" * 60)
        print("PHASE 1: DOCUMENT PROCESSING BENCHMARK")
        print("=" * 60)
        
        results = []
        pdf_files = list(self.test_dir.glob("*.pdf"))
        
        for pdf_path in pdf_files:
            print(f"\n📄 Processing: {pdf_path.name}")
            
            # Create a fake uploaded file object
            class FakeUploadedFile:
                def __init__(self, path):
                    self.name = path.name
                    self.size = path.stat().st_size
                    self._path = path
                
                def getbuffer(self):
                    with open(self._path, 'rb') as f:
                        return f.read()
            
            fake_file = FakeUploadedFile(pdf_path)
            
            # Measure processing time
            start_time = time.perf_counter()
            
            try:
                result = session.add_documents([fake_file])
                processing_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
                
                if result and result[0].get('success'):
                    doc_result = {
                        'filename': pdf_path.name,
                        'size_kb': pdf_path.stat().st_size / 1024,
                        'chunks': result[0].get('chunks', 0),
                        'processing_time_ms': round(processing_time, 2),
                        'status': 'SUCCESS'
                    }
                    print(f"   ✅ {result[0].get('chunks', 0)} chunks in {processing_time:.2f}ms")
                else:
                    doc_result = {
                        'filename': pdf_path.name,
                        'size_kb': pdf_path.stat().st_size / 1024,
                        'chunks': 0,
                        'processing_time_ms': round(processing_time, 2),
                        'status': 'FAILED',
                        'error': result[0].get('error', 'Unknown error') if result else 'No result'
                    }
                    print(f"   ❌ Failed: {doc_result.get('error')}")
                
                results.append(doc_result)
                
            except Exception as e:
                processing_time = (time.perf_counter() - start_time) * 1000
                results.append({
                    'filename': pdf_path.name,
                    'size_kb': pdf_path.stat().st_size / 1024,
                    'chunks': 0,
                    'processing_time_ms': round(processing_time, 2),
                    'status': 'ERROR',
                    'error': str(e)
                })
                print(f"   ❌ Error: {str(e)}")
        
        return results
    
    def benchmark_queries(self, session: RAGSession) -> Tuple[List[Dict], List[Dict]]:
        """
        Benchmark query performance and accuracy.
        
        Args:
            session: RAGSession instance with documents loaded.
            
        Returns:
            Tuple of (performance_results, accuracy_results)
        """
        print("\n" + "=" * 60)
        print("PHASE 2: QUERY PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        performance_results = []
        accuracy_results = []
        
        # Ensure LLM is loaded (measure separately)
        print("\n⏳ Loading LLM model...")
        model_load_start = time.perf_counter()
        session.load_llm()
        model_load_time = (time.perf_counter() - model_load_start) * 1000
        print(f"✅ Model loaded in {model_load_time:.2f}ms")
        
        # Warm up the model
        print("🔥 Warming up model...")
        session.warmup_model()
        
        query_count = 0
        for doc_name, qa_pairs in self.ground_truth.items():
            # Find matching document
            matching_docs = [d for d in session.documents if doc_name in d['filename'] or d['filename'] in doc_name]
            
            if not matching_docs:
                print(f"\n⚠️ Document not found in session: {doc_name}")
                continue
            
            actual_doc_name = matching_docs[0]['filename']
            print(f"\n📄 Testing: {actual_doc_name}")
            
            # Get questions and answers
            questions = [(k, v) for k, v in qa_pairs.items() if k.startswith('Question')]
            answers = {k: v for k, v in qa_pairs.items() if k.startswith('Answer')}
            
            for q_key, question in questions:
                query_count += 1
                q_num = q_key.split()[-1]
                expected_answer = answers.get(f'Answer {q_num}', '')
                
                print(f"\n   Q{q_num}: {question[:60]}...")
                
                # Measure retrieval time
                retrieval_start = time.perf_counter()
                chunks = session.retrieve_context(question, selected_document=actual_doc_name)
                retrieval_time = (time.perf_counter() - retrieval_start) * 1000
                
                # Measure inference time
                inference_start = time.perf_counter()
                response, debug_info = session.generate_response(
                    question,
                    compliance_mode=False,
                    selected_document=actual_doc_name
                )
                inference_time = (time.perf_counter() - inference_start) * 1000
                
                total_time = retrieval_time + inference_time
                
                # Calculate accuracy
                accuracy = self._calculate_accuracy(response, expected_answer)
                
                # Store performance results
                perf_result = {
                    'query_id': query_count,
                    'document': actual_doc_name,
                    'question': question,
                    'retrieval_time_ms': round(retrieval_time, 2),
                    'inference_time_ms': round(inference_time, 2),
                    'total_time_ms': round(total_time, 2),
                    'chunks_retrieved': len(chunks),
                    'input_tokens': debug_info.get('input_tokens', 0),
                    'output_tokens': debug_info.get('output_tokens', 0)
                }
                performance_results.append(perf_result)
                
                # Store accuracy results
                acc_result = {
                    'query_id': query_count,
                    'document': actual_doc_name,
                    'question': question,
                    'expected_answer': expected_answer,
                    'actual_response': response[:500],  # Truncate for readability
                    'accuracy_score': round(accuracy, 2),
                    'is_correct': accuracy >= 0.5
                }
                accuracy_results.append(acc_result)
                
                # Print results
                status = "✅" if accuracy >= 0.5 else "⚠️"
                print(f"       Retrieval: {retrieval_time:.2f}ms | Inference: {inference_time:.2f}ms | Total: {total_time:.2f}ms")
                print(f"       {status} Accuracy: {accuracy*100:.0f}%")
                print(f"       Expected: {expected_answer[:50]}...")
                print(f"       Got: {response[:50]}...")
        
        return performance_results, accuracy_results
    
    def run_benchmark(self) -> Dict:
        """
        Run the complete benchmark suite.
        
        Returns:
            Dictionary containing all benchmark results.
        """
        print("\n" + "=" * 70)
        print("   RAG SYSTEM PERFORMANCE BENCHMARK")
        print("=" * 70)
        
        # Collect system info
        print("\n📊 Collecting system information...")
        self.results['system_info'] = self._get_system_info()
        
        print(f"\n🖥️  System: {self.results['system_info']['platform']}")
        print(f"💻 CPU: {self.results['system_info']['processor']} ({self.results['system_info']['cpu_count']} cores)")
        print(f"🎮 GPU: {self.results['system_info']['gpu_name']}")
        print(f"💾 RAM: {self.results['system_info']['ram_total']}")
        print(f"🔥 CUDA: {'Available' if self.results['system_info']['cuda_available'] else 'Not Available'}")
        
        # Create benchmark session
        print("\n📁 Creating benchmark session...")
        session_id = self.session_manager.create_session("Benchmark_Session")
        session = self.session_manager.get_session(session_id)
        
        try:
            # Phase 1: Document Processing
            self.results['document_processing'] = self.benchmark_document_processing(session)
            
            # Phase 2: Query Performance
            perf_results, acc_results = self.benchmark_queries(session)
            self.results['query_performance'] = perf_results
            self.results['accuracy_results'] = acc_results
            
            # Calculate summary statistics
            self.results['summary'] = self._calculate_summary()
            
        finally:
            # Cleanup
            print("\n🧹 Cleaning up benchmark session...")
            self.session_manager.delete_session(session_id)
        
        return self.results
    
    def _calculate_summary(self) -> Dict:
        """Calculate summary statistics from benchmark results."""
        summary = {}
        
        # Document processing summary
        if self.results['document_processing']:
            proc_times = [d['processing_time_ms'] for d in self.results['document_processing'] if d['status'] == 'SUCCESS']
            total_chunks = sum(d['chunks'] for d in self.results['document_processing'])
            
            summary['document_processing'] = {
                'total_documents': len(self.results['document_processing']),
                'successful': len(proc_times),
                'total_chunks': total_chunks,
                'avg_processing_time_ms': round(statistics.mean(proc_times), 2) if proc_times else 0,
                'total_processing_time_ms': round(sum(proc_times), 2) if proc_times else 0
            }
        
        # Query performance summary
        if self.results['query_performance']:
            retrieval_times = [q['retrieval_time_ms'] for q in self.results['query_performance']]
            inference_times = [q['inference_time_ms'] for q in self.results['query_performance']]
            total_times = [q['total_time_ms'] for q in self.results['query_performance']]
            
            summary['query_performance'] = {
                'total_queries': len(self.results['query_performance']),
                'avg_retrieval_ms': round(statistics.mean(retrieval_times), 2),
                'min_retrieval_ms': round(min(retrieval_times), 2),
                'max_retrieval_ms': round(max(retrieval_times), 2),
                'avg_inference_ms': round(statistics.mean(inference_times), 2),
                'min_inference_ms': round(min(inference_times), 2),
                'max_inference_ms': round(max(inference_times), 2),
                'avg_total_ms': round(statistics.mean(total_times), 2),
                'min_total_ms': round(min(total_times), 2),
                'max_total_ms': round(max(total_times), 2)
            }
        
        # Accuracy summary
        if self.results['accuracy_results']:
            accuracies = [a['accuracy_score'] for a in self.results['accuracy_results']]
            correct = sum(1 for a in self.results['accuracy_results'] if a['is_correct'])
            
            summary['accuracy'] = {
                'total_questions': len(accuracies),
                'correct_answers': correct,
                'accuracy_rate': round(correct / len(accuracies) * 100, 1),
                'avg_accuracy_score': round(statistics.mean(accuracies) * 100, 1),
                'min_accuracy_score': round(min(accuracies) * 100, 1),
                'max_accuracy_score': round(max(accuracies) * 100, 1)
            }
        
        return summary
    
    def generate_report(self, output_path: str = None) -> str:
        """
        Generate a markdown report of the benchmark results.
        
        Args:
            output_path: Path to save the report (optional).
            
        Returns:
            Markdown report string.
        """
        report = []
        report.append("# RAG System Performance Benchmark Report\n")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n")
        
        # System Information
        report.append("## 1. System Information\n")
        report.append("| Component | Value |")
        report.append("|-----------|-------|")
        sys_info = self.results['system_info']
        report.append(f"| Platform | {sys_info.get('platform', 'N/A')} |")
        report.append(f"| CPU | {sys_info.get('processor', 'N/A')} ({sys_info.get('cpu_count', 'N/A')} cores) |")
        report.append(f"| RAM | {sys_info.get('ram_total', 'N/A')} |")
        report.append(f"| GPU | {sys_info.get('gpu_name', 'N/A')} |")
        report.append(f"| GPU Memory | {sys_info.get('gpu_memory', 'N/A')} |")
        report.append(f"| CUDA | {'Available' if sys_info.get('cuda_available') else 'Not Available'} |")
        report.append(f"| PyTorch | {sys_info.get('torch_version', 'N/A')} |")
        report.append("")
        
        # Document Processing Results
        report.append("## 2. Document Processing\n")
        if self.results['document_processing']:
            report.append("| Document | Size (KB) | Chunks | Time (ms) | Status |")
            report.append("|----------|-----------|--------|-----------|--------|")
            for doc in self.results['document_processing']:
                status_icon = "✅" if doc['status'] == 'SUCCESS' else "❌"
                report.append(f"| {doc['filename'][:40]}... | {doc['size_kb']:.1f} | {doc['chunks']} | {doc['processing_time_ms']:.0f} | {status_icon} |")
            report.append("")
            
            summary = self.results['summary'].get('document_processing', {})
            report.append(f"**Total Documents:** {summary.get('total_documents', 0)} | ")
            report.append(f"**Total Chunks:** {summary.get('total_chunks', 0)} | ")
            report.append(f"**Avg Processing Time:** {summary.get('avg_processing_time_ms', 0):.0f}ms\n")
        
        # Query Performance Results
        report.append("## 3. Query Performance\n")
        if self.results['query_performance']:
            report.append("| Query | Retrieval (ms) | Inference (ms) | Total (ms) | Chunks |")
            report.append("|-------|----------------|----------------|------------|--------|")
            for q in self.results['query_performance']:
                report.append(f"| Q{q['query_id']} | {q['retrieval_time_ms']:.1f} | {q['inference_time_ms']:.1f} | **{q['total_time_ms']:.0f}** | {q['chunks_retrieved']} |")
            report.append("")
            
            summary = self.results['summary'].get('query_performance', {})
            report.append("### Performance Summary\n")
            report.append("| Metric | Retrieval | Inference | Total |")
            report.append("|--------|-----------|-----------|-------|")
            report.append(f"| Average | {summary.get('avg_retrieval_ms', 0):.1f}ms | {summary.get('avg_inference_ms', 0):.1f}ms | **{summary.get('avg_total_ms', 0):.0f}ms** |")
            report.append(f"| Min | {summary.get('min_retrieval_ms', 0):.1f}ms | {summary.get('min_inference_ms', 0):.1f}ms | {summary.get('min_total_ms', 0):.0f}ms |")
            report.append(f"| Max | {summary.get('max_retrieval_ms', 0):.1f}ms | {summary.get('max_inference_ms', 0):.1f}ms | {summary.get('max_total_ms', 0):.0f}ms |")
            report.append("")
        
        # Accuracy Results
        report.append("## 4. Answer Accuracy\n")
        if self.results['accuracy_results']:
            report.append("| Query | Document | Accuracy | Status |")
            report.append("|-------|----------|----------|--------|")
            for a in self.results['accuracy_results']:
                status = "✅ CORRECT" if a['is_correct'] else "⚠️ PARTIAL"
                doc_short = a['document'][:25] + "..." if len(a['document']) > 25 else a['document']
                report.append(f"| Q{a['query_id']} | {doc_short} | {a['accuracy_score']*100:.0f}% | {status} |")
            report.append("")
            
            acc_summary = self.results['summary'].get('accuracy', {})
            report.append("### Accuracy Summary\n")
            report.append(f"- **Total Questions:** {acc_summary.get('total_questions', 0)}")
            report.append(f"- **Correct Answers:** {acc_summary.get('correct_answers', 0)}")
            report.append(f"- **Accuracy Rate:** {acc_summary.get('accuracy_rate', 0):.1f}%")
            report.append(f"- **Avg Score:** {acc_summary.get('avg_accuracy_score', 0):.1f}%")
            report.append("")
        
        # Key Findings
        report.append("## 5. Key Findings\n")
        if self.results['summary']:
            perf = self.results['summary'].get('query_performance', {})
            acc = self.results['summary'].get('accuracy', {})
            
            report.append(f"1. **Retrieval Performance:** Average {perf.get('avg_retrieval_ms', 0):.1f}ms (vector search is fast)")
            report.append(f"2. **Inference Performance:** Average {perf.get('avg_inference_ms', 0):.1f}ms (LLM generation is the bottleneck)")
            report.append(f"3. **Total Response Time:** Average {perf.get('avg_total_ms', 0):.0f}ms per query")
            report.append(f"4. **Answer Accuracy:** {acc.get('accuracy_rate', 0):.1f}% of answers are correct")
            report.append("")
            
            # Thesis citation
            report.append("### Thesis Citation\n")
            report.append(f"> \"The RAG system demonstrates an average retrieval time of {perf.get('avg_retrieval_ms', 0):.1f}ms ")
            report.append(f"and an average inference time of {perf.get('avg_inference_ms', 0)/1000:.2f}s, ")
            report.append(f"resulting in a total average response time of {perf.get('avg_total_ms', 0)/1000:.2f}s. ")
            report.append(f"When tested against ground truth questions from procurement documents, ")
            report.append(f"the system achieved an accuracy rate of {acc.get('accuracy_rate', 0):.1f}%.\"")
        
        report_text = "\n".join(report)
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n📄 Report saved to: {output_path}")
        
        return report_text


def main():
    """Run the benchmark and generate report."""
    print("\n" + "=" * 70)
    print("   RAG SYSTEM PERFORMANCE BENCHMARK")
    print("   Testing Retrieval & Inference with Ground Truth Data")
    print("=" * 70)
    
    # Initialize benchmark
    benchmark = RAGBenchmark(test_dir="./test")
    
    # Run benchmark
    results = benchmark.run_benchmark()
    
    # Generate and save report
    report_path = Path("./reports/benchmark_performance_report.md")
    report_path.parent.mkdir(exist_ok=True)
    
    report = benchmark.generate_report(str(report_path))
    
    # Also save JSON results
    json_path = report_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📊 JSON results saved to: {json_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("   BENCHMARK COMPLETE")
    print("=" * 70)
    
    summary = results.get('summary', {})
    perf = summary.get('query_performance', {})
    acc = summary.get('accuracy', {})
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print(f"   Average Retrieval Time: {perf.get('avg_retrieval_ms', 0):.1f}ms")
    print(f"   Average Inference Time: {perf.get('avg_inference_ms', 0):.1f}ms")
    print(f"   Average Total Time:     {perf.get('avg_total_ms', 0):.0f}ms")
    
    print(f"\n🎯 ACCURACY SUMMARY:")
    print(f"   Total Questions: {acc.get('total_questions', 0)}")
    print(f"   Correct Answers: {acc.get('correct_answers', 0)}")
    print(f"   Accuracy Rate:   {acc.get('accuracy_rate', 0):.1f}%")
    
    print(f"\n📄 Full report: {report_path}")
    print(f"📊 JSON data:   {json_path}")


if __name__ == "__main__":
    main()
