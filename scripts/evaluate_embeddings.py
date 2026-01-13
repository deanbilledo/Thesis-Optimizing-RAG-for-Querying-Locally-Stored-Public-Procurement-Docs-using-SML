#!/usr/bin/env python3
"""
Evaluate embedding models for retrieval using Recall@K
"""

from sentence_transformers import SentenceTransformer
import json
import numpy as np
from tqdm import tqdm
import argparse
import time

def load_embeddings(path):
    """Load document embeddings from JSON"""
    print(f"📂 Loading embeddings from: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    embeddings_dict = {d['id']: np.array(d['embedding']) for d in data}
    print(f"✅ Loaded {len(embeddings_dict)} document embeddings")
    return embeddings_dict

def load_qa_pairs(path):
    """Load Q&A pairs from JSONL"""
    print(f"📂 Loading Q&A pairs from: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        qa_data = [json.loads(line) for line in f if line.strip()]
    print(f"✅ Loaded {len(qa_data)} Q&A pairs")
    return qa_data

def evaluate(model_name, embedding_file, qa_file, k_values=[1, 3, 5, 10]):
    """
    Evaluate retrieval performance using Recall@K
    
    Args:
        model_name: HuggingFace model name
        embedding_file: Path to document embeddings JSON
        qa_file: Path to qa_pairs.jsonl
        k_values: List of K values to evaluate
    
    Returns:
        Dictionary of recall scores
    """
    print(f"\n{'='*70}")
    print(f"🔍 EVALUATING: {model_name}")
    print(f"{'='*70}\n")
    
    # Load model
    print(f"🔄 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Load embeddings and Q&A pairs
    doc_embeddings = load_embeddings(embedding_file)
    qa_data = load_qa_pairs(qa_file)
    
    # Create document matrix
    doc_ids = list(doc_embeddings.keys())
    doc_matrix = np.vstack([doc_embeddings[doc_id] for doc_id in doc_ids])
    print(f"📊 Document matrix shape: {doc_matrix.shape}")

    # Initialize counters
    recall_counters = {k: 0 for k in k_values}
    total = 0
    query_times = []

    print(f"\n🎯 Evaluating queries...")
    for item in tqdm(qa_data, desc="Processing"):
        query = item['query']
        gold_id = item['gold_document_id']

        # Encode query and measure time
        start_time = time.time()
        query_emb = model.encode(query)
        query_time = (time.time() - start_time) * 1000  # Convert to ms
        query_times.append(query_time)
        
        # Compute similarity scores
        scores = np.dot(doc_matrix, query_emb)
        
        # Get top-K for each K value
        max_k = max(k_values)
        top_k_indices = np.argsort(scores)[::-1][:max_k]
        
        # Check recall for each K
        for k in k_values:
            top_k_ids = [doc_ids[i] for i in top_k_indices[:k]]
            if gold_id in top_k_ids:
                recall_counters[k] += 1
        
        total += 1

    # Calculate metrics
    results = {
        "model": model_name,
        "total_queries": total,
        "avg_query_time_ms": np.mean(query_times),
        "median_query_time_ms": np.median(query_times),
    }
    
    for k in k_values:
        recall_score = recall_counters[k] / total if total > 0 else 0
        results[f"recall@{k}"] = recall_score
    
    return results

def print_results(results):
    """Pretty print evaluation results"""
    print(f"\n{'='*70}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*70}\n")
    
    print(f"Model: {results['model']}")
    print(f"Total Queries: {results['total_queries']}")
    print(f"Avg Query Time: {results['avg_query_time_ms']:.2f} ms")
    print(f"Median Query Time: {results['median_query_time_ms']:.2f} ms")
    print(f"\nRetrieval Performance:")
    
    for key, value in results.items():
        if key.startswith("recall@"):
            k = key.split("@")[1]
            print(f"  Recall@{k}: {value:.4f} ({value*100:.2f}%)")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate embedding model for retrieval"
    )
    parser.add_argument(
        "--model", 
        required=True,
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--embeddings", 
        required=True,
        help="Path to embeddings JSON file"
    )
    parser.add_argument(
        "--qa", 
        default="qa_pairs.jsonl",
        help="Path to qa_pairs.jsonl"
    )
    parser.add_argument(
        "--k", 
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="K values for Recall@K (default: 1 3 5 10)"
    )
    args = parser.parse_args()

    results = evaluate(args.model, args.embeddings, args.qa, k_values=args.k)
    print_results(results)
    
    # Save results
    output_file = args.embeddings.replace(".json", "_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")
