#!/usr/bin/env python3
"""
Master script to run complete embedding evaluation pipeline
"""

import subprocess
import json
import pandas as pd
import os

# ================= CONFIGURATION =================
MODELS_TO_EVALUATE = [
    "all-MiniLM-L6-v2",
    "intfloat/e5-small-v2",
    "all-mpnet-base-v2",  # FIXED: Correct model name
]

DOCUMENTS_FILE = "documents.jsonl"
QA_FILE = "qa_pairs.jsonl"
RESULTS_DIR = "results"


def run_command(cmd):
    """Run shell command and print output"""
    print(f"\n🔄 Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"⚠️  Command failed with return code {result.returncode}")
        return False
    return True


def main():
    print("=" * 70)
    print("🚀 EMBEDDING MODEL EVALUATION PIPELINE")
    print("=" * 70)
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Step 1: Prepare dataset
    print(f"\n📚 STEP 1: Preparing dataset...")
    if not os.path.exists(DOCUMENTS_FILE) or not os.path.exists(QA_FILE):
        print(f"⏳ Running dataset preparation...")
        if not run_command(["python", "prepare_retrieval_dataset.py"]):
            print("❌ Dataset preparation failed!")
            return
    else:
        print(f"✅ Dataset files already exist")
    
    # Step 2: Encode corpus for each model
    print(f"\n🧠 STEP 2: Encoding corpus with {len(MODELS_TO_EVALUATE)} models...")
    for model in MODELS_TO_EVALUATE:
        model_name_safe = model.replace("/", "_")
        output_file = f"{RESULTS_DIR}/{model_name_safe}_embeddings.json"
        
        if os.path.exists(output_file):
            print(f"✅ Embeddings already exist for {model}")
            continue
        
        cmd = [
            "python", "scripts/encode_corpus.py",
            "--model", model,
            "--docs", DOCUMENTS_FILE,
            "--output", output_file
        ]
        
        if not run_command(cmd):
            print(f"⚠️  Encoding failed for {model}, skipping...")
            continue
    
    # Step 3: Evaluate each model
    print(f"\n🎯 STEP 3: Evaluating retrieval performance...")
    all_results = []
    
    for model in MODELS_TO_EVALUATE:
        model_name_safe = model.replace("/", "_")
        embeddings_file = f"{RESULTS_DIR}/{model_name_safe}_embeddings.json"
        
        if not os.path.exists(embeddings_file):
            print(f"⚠️  Embeddings not found for {model}, skipping...")
            continue
        
        cmd = [
            "python", "scripts/evaluate_embeddings.py",
            "--model", model,
            "--embeddings", embeddings_file,
            "--qa", QA_FILE,
            "--k", "1", "3", "5", "10"
        ]
        
        if run_command(cmd):
            # Load results
            results_file = embeddings_file.replace(".json", "_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    all_results.append(results)
    
    # Step 4: Create comparison table
    print(f"\n📊 STEP 4: Creating comparison table...")
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Select columns for display
        display_cols = ['model', 'recall@1', 'recall@3', 'recall@5', 'recall@10', 
                       'avg_query_time_ms', 'median_query_time_ms']
        df_display = df[display_cols]
        
        print(f"\n{'='*70}")
        print("📈 EMBEDDING MODEL COMPARISON")
        print(f"{'='*70}\n")
        print(df_display.to_string(index=False))
        
        # Save to CSV
        csv_file = f"{RESULTS_DIR}/embedding_comparison.csv"
        df_display.to_csv(csv_file, index=False)
        print(f"\n💾 Comparison table saved to: {csv_file}")
        
        # Find best model
        best_recall5 = df.loc[df['recall@5'].idxmax()]
        best_speed = df.loc[df['avg_query_time_ms'].idxmin()]
        
        print(f"\n🏆 RECOMMENDATIONS:")
        print(f"  Best Accuracy: {best_recall5['model']} (Recall@5: {best_recall5['recall@5']:.4f})")
        print(f"  Fastest: {best_speed['model']} ({best_speed['avg_query_time_ms']:.2f} ms/query)")
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION PIPELINE COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 All results saved in: {RESULTS_DIR}/")
    print(f"\n🚀 Next step: Run end-to-end RAG evaluation:")
    print(f"   python scripts/evaluate_rag.py \\")
    print(f"       --embedding-model <best_model> \\")
    print(f"       --embeddings results/<model>_embeddings.json \\")
    print(f"       --llm meta-llama/Llama-3.2-1B-Instruct \\")
    print(f"       --adapter ./llama_1b_procurement_weska_v2")


if __name__ == "__main__":
    main()
