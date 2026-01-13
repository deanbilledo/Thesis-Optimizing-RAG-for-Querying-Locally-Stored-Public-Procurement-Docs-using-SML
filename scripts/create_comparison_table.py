#!/usr/bin/env python3
"""
Generate comparison tables and visualizations from evaluation results
"""

import json
import pandas as pd
import os
from pathlib import Path

RESULTS_DIR = "results"

def load_all_results():
    """Load all evaluation results from JSON files"""
    results = []
    results_path = Path(RESULTS_DIR)
    
    for file in results_path.glob("*_results.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results

def create_comparison_table(results):
    """Create comprehensive comparison table"""
    if not results:
        print("⚠️  No results found!")
        return None
    
    df = pd.DataFrame(results)
    
    # Format model names (shorten for display)
    df['model_short'] = df['model'].apply(lambda x: x.split('/')[-1])
    
    # Select and order columns
    cols = ['model_short', 'recall@1', 'recall@3', 'recall@5', 'recall@10', 
            'avg_query_time_ms', 'median_query_time_ms', 'total_queries']
    
    df_display = df[cols].copy()
    
    # Round numeric columns
    numeric_cols = ['recall@1', 'recall@3', 'recall@5', 'recall@10', 
                   'avg_query_time_ms', 'median_query_time_ms']
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(4)
    
    # Rename columns for better display
    df_display = df_display.rename(columns={
        'model_short': 'Model',
        'recall@1': 'R@1',
        'recall@3': 'R@3',
        'recall@5': 'R@5',
        'recall@10': 'R@10',
        'avg_query_time_ms': 'Avg Time (ms)',
        'median_query_time_ms': 'Med Time (ms)',
        'total_queries': 'Queries'
    })
    
    return df_display

def print_summary(df):
    """Print summary and recommendations"""
    print(f"\n{'='*80}")
    print("📊 EMBEDDING MODEL COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    print(df.to_string(index=False))
    
    # Find best performers
    if 'R@5' in df.columns:
        best_accuracy_idx = df['R@5'].idxmax()
        best_accuracy = df.iloc[best_accuracy_idx]
        
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"\n  📈 Highest Accuracy (R@5):")
        print(f"     Model: {best_accuracy['Model']}")
        print(f"     Recall@5: {best_accuracy['R@5']:.4f} ({best_accuracy['R@5']*100:.2f}%)")
        print(f"     Avg Time: {best_accuracy['Avg Time (ms)']:.2f} ms")
    
    if 'Avg Time (ms)' in df.columns:
        fastest_idx = df['Avg Time (ms)'].idxmin()
        fastest = df.iloc[fastest_idx]
        
        print(f"\n  ⚡ Fastest:")
        print(f"     Model: {fastest['Model']}")
        print(f"     Avg Time: {fastest['Avg Time (ms)']:.2f} ms")
        print(f"     Recall@5: {fastest['R@5']:.4f} ({fastest['R@5']*100:.2f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"\n  Production Use:")
    if 'R@5' in df.columns and 'Avg Time (ms)' in df.columns:
        # Find balanced model
        df_temp = df.copy()
        df_temp['score'] = df_temp['R@5'] * 0.7 + (1 - df_temp['Avg Time (ms)'] / df_temp['Avg Time (ms)'].max()) * 0.3
        balanced_idx = df_temp['score'].idxmax()
        balanced = df.iloc[balanced_idx]
        
        print(f"    • High Accuracy: {best_accuracy['Model']}")
        print(f"    • Balanced: {balanced['Model']}")
        print(f"    • Speed-Critical: {fastest['Model']}")
    
    print(f"\n{'='*80}")

def create_markdown_table(df):
    """Create markdown-formatted table"""
    md = "# Embedding Model Comparison Results\n\n"
    md += "## Performance Metrics\n\n"
    md += df.to_markdown(index=False)
    md += "\n\n## Legend\n\n"
    md += "- **R@K**: Recall at K (percentage of queries with gold document in top-K)\n"
    md += "- **Avg Time**: Average query encoding time in milliseconds\n"
    md += "- **Med Time**: Median query encoding time in milliseconds\n"
    
    return md

def main():
    print("📊 Loading evaluation results...")
    
    # Load results
    results = load_all_results()
    
    if not results:
        print("❌ No results found in results/ directory!")
        print("💡 Run the evaluation pipeline first: run_embedding_eval.bat")
        return
    
    print(f"✅ Found {len(results)} evaluation results")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    if df is None:
        return
    
    # Print summary
    print_summary(df)
    
    # Save to CSV
    csv_file = f"{RESULTS_DIR}/embedding_comparison.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n💾 CSV saved to: {csv_file}")
    
    # Save to Markdown
    md_file = f"{RESULTS_DIR}/embedding_comparison.md"
    md_content = create_markdown_table(df)
    with open(md_file, 'w') as f:
        f.write(md_content)
    print(f"💾 Markdown saved to: {md_file}")
    
    print(f"\n✅ Comparison complete!")

if __name__ == "__main__":
    main()
