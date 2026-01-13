#!/usr/bin/env python3
"""
Complete End-to-End RAG Evaluation Pipeline
Tests all embedding models with fine-tuned LLaMA 3.2-1B
"""

import subprocess
import json
import os
from pathlib import Path
import time

# === CONFIGURATION ===
MODELS_TO_EVALUATE = [
    {
        "name": "all-MiniLM-L6-v2",
        "embeddings": "results/all-MiniLM-L6-v2_embeddings.json",
        "description": "Fast & Lightweight (384 dim)"
    },
    {
        "name": "intfloat/e5-small-v2",
        "embeddings": "results/intfloat_e5-small-v2_embeddings.json",
        "description": "Balanced Performance (384 dim)"
    },
    {
        "name": "all-mpnet-base-v2",
        "embeddings": "results/all-mpnet-base-v2_embeddings.json",
        "description": "High Accuracy (768 dim)"
    }
]

LLM_CONFIG = {
    "base_model": "meta-llama/Llama-3.2-1B-Instruct",
    "adapter": "./llama_1b_procurement_weska_v2",
    "top_k": 5,
    "max_tokens": 200
}

DATASET_CONFIG = {
    "documents": "documents.jsonl",
    "qa_pairs": "qa_pairs.jsonl"
}

RESULTS_DIR = "results/rag"

# === MAIN EXECUTION ===
def ensure_directories():
    """Ensure results directories exist"""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

def check_embeddings_exist():
    """Check if all embedding files exist"""
    missing = []
    for model in MODELS_TO_EVALUATE:
        if not os.path.exists(model["embeddings"]):
            missing.append(model["name"])
    
    if missing:
        print(f"⚠️  Missing embeddings for: {', '.join(missing)}")
        print(f"💡 Run the embedding pipeline first: run_embedding_eval.bat")
        return False
    return True

def run_rag_evaluation(model_config):
    """Run RAG evaluation for a single embedding model"""
    model_name = model_config["name"]
    embeddings_file = model_config["embeddings"]
    
    print(f"\n{'='*70}")
    print(f"🚀 EVALUATING: {model_name}")
    print(f"   {model_config['description']}")
    print(f"{'='*70}\n")
    
    # Sanitize model name for filename
    safe_name = model_name.replace('/', '_')
    output_file = f"{RESULTS_DIR}/{safe_name}_rag_results.json"
    
    # Build command
    cmd = [
        "python", "scripts/evaluate_rag.py",
        "--embedding-model", model_name,
        "--embeddings", embeddings_file,
        "--documents", DATASET_CONFIG["documents"],
        "--qa", DATASET_CONFIG["qa_pairs"],
        "--llm", LLM_CONFIG["base_model"],
        "--adapter", LLM_CONFIG["adapter"],
        "--top-k", str(LLM_CONFIG["top_k"]),
        "--max-tokens", str(LLM_CONFIG["max_tokens"]),
        "--output", output_file
    ]
    
    # Run evaluation
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  Completed in {elapsed_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error evaluating {model_name}:")
        print(e.stderr)
        return False

def create_rag_comparison_table():
    """Create comparison table from all RAG results"""
    import pandas as pd
    
    print(f"\n{'='*70}")
    print(f"📊 CREATING RAG COMPARISON TABLE")
    print(f"{'='*70}\n")
    
    results = []
    results_path = Path(RESULTS_DIR)
    
    for file in results_path.glob("*_rag_results.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    if not results:
        print("⚠️  No RAG results found!")
        return
    
    df = pd.DataFrame(results)
    
    # Select columns for display
    cols = [
        'embedding_model',
        'retrieval_recall',
        'avg_bleu',
        'avg_rouge1',
        'avg_rouge2',
        'avg_rougeL',
        'avg_f1',
        'avg_exact_match'
    ]
    
    df_display = df[cols].copy()
    
    # Rename for better display
    df_display = df_display.rename(columns={
        'embedding_model': 'Embedding Model',
        'retrieval_recall': 'Retrieval R@5',
        'avg_bleu': 'BLEU',
        'avg_rouge1': 'ROUGE-1',
        'avg_rouge2': 'ROUGE-2',
        'avg_rougeL': 'ROUGE-L',
        'avg_f1': 'F1',
        'avg_exact_match': 'Exact Match'
    })
    
    # Round numeric columns
    numeric_cols = ['Retrieval R@5', 'BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'F1', 'Exact Match']
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(4)
    
    # Print summary
    print(df_display.to_string(index=False))
    
    # Save to CSV
    csv_file = f"{RESULTS_DIR}/rag_comparison.csv"
    df_display.to_csv(csv_file, index=False)
    print(f"\n💾 CSV saved to: {csv_file}")
    
    # Save to Markdown
    md_file = f"{RESULTS_DIR}/rag_comparison.md"
    with open(md_file, 'w') as f:
        f.write("# End-to-End RAG Evaluation Results\n\n")
        f.write(f"**LLM**: {LLM_CONFIG['base_model']}\n")
        f.write(f"**Adapter**: {LLM_CONFIG['adapter']}\n")
        f.write(f"**Top-K**: {LLM_CONFIG['top_k']}\n\n")
        f.write("## Results\n\n")
        f.write(df_display.to_markdown(index=False))
        f.write("\n\n## Metrics\n\n")
        f.write("- **Retrieval R@5**: Percentage of queries where gold document is in top-5\n")
        f.write("- **BLEU**: N-gram overlap score (0-1, higher is better)\n")
        f.write("- **ROUGE-1/2/L**: Unigram/Bigram/Longest-common-subsequence overlap\n")
        f.write("- **F1**: Token-level F1 score\n")
        f.write("- **Exact Match**: Percentage of exact string matches\n")
    
    print(f"💾 Markdown saved to: {md_file}")
    
    # Print recommendations
    print(f"\n{'='*70}")
    print(f"🏆 BEST PERFORMERS:")
    print(f"{'='*70}\n")
    
    best_retrieval = df_display.loc[df_display['Retrieval R@5'].idxmax()]
    print(f"📈 Best Retrieval: {best_retrieval['Embedding Model']}")
    print(f"   R@5: {best_retrieval['Retrieval R@5']:.4f} ({best_retrieval['Retrieval R@5']*100:.2f}%)")
    
    best_rouge = df_display.loc[df_display['ROUGE-L'].idxmax()]
    print(f"\n📝 Best Answer Quality (ROUGE-L): {best_rouge['Embedding Model']}")
    print(f"   ROUGE-L: {best_rouge['ROUGE-L']:.4f}")
    print(f"   F1: {best_rouge['F1']:.4f}")
    
    best_bleu = df_display.loc[df_display['BLEU'].idxmax()]
    print(f"\n🎯 Best BLEU: {best_bleu['Embedding Model']}")
    print(f"   BLEU: {best_bleu['BLEU']:.4f}")
    
    print(f"\n{'='*70}")

def main():
    print(f"\n{'='*70}")
    print(f"🚀 END-TO-END RAG EVALUATION PIPELINE")
    print(f"{'='*70}\n")
    
    print(f"LLM: {LLM_CONFIG['base_model']}")
    print(f"Adapter: {LLM_CONFIG['adapter']}")
    print(f"Top-K: {LLM_CONFIG['top_k']}")
    print(f"Embedding Models: {len(MODELS_TO_EVALUATE)}")
    
    # Check prerequisites
    ensure_directories()
    
    if not check_embeddings_exist():
        return
    
    if not os.path.exists(DATASET_CONFIG["documents"]):
        print(f"❌ Missing {DATASET_CONFIG['documents']}")
        print(f"💡 Run: python prepare_retrieval_dataset.py")
        return
    
    if not os.path.exists(DATASET_CONFIG["qa_pairs"]):
        print(f"❌ Missing {DATASET_CONFIG['qa_pairs']}")
        print(f"💡 Run: python prepare_retrieval_dataset.py")
        return
    
    # Run evaluations
    total_start = time.time()
    success_count = 0
    
    for model_config in MODELS_TO_EVALUATE:
        if run_rag_evaluation(model_config):
            success_count += 1
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"✅ Completed {success_count}/{len(MODELS_TO_EVALUATE)} evaluations")
    print(f"⏱️  Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    print(f"{'='*70}")
    
    # Create comparison table
    if success_count > 0:
        create_rag_comparison_table()
    
    print(f"\n🎉 END-TO-END RAG EVALUATION COMPLETE!")
    print(f"📂 Results saved in: {RESULTS_DIR}/")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
