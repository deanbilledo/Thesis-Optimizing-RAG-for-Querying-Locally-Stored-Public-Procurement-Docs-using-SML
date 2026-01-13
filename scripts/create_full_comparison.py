#!/usr/bin/env python3
"""
Generate comprehensive comparison tables for:
1. Embedding-only evaluation (Recall@K)
2. End-to-end RAG evaluation (BLEU, ROUGE, F1)
3. Combined view
"""

import json
import pandas as pd
import os
from pathlib import Path

RESULTS_DIR = "results"
RAG_RESULTS_DIR = "results/rag"

def load_embedding_results():
    """Load embedding evaluation results"""
    results = []
    results_path = Path(RESULTS_DIR)
    
    for file in results_path.glob("*_results.json"):
        # Skip RAG results
        if "rag" not in str(file):
            with open(file, 'r') as f:
                data = json.load(f)
                results.append(data)
    
    return results

def load_rag_results():
    """Load RAG evaluation results"""
    results = []
    rag_path = Path(RAG_RESULTS_DIR)
    
    if not rag_path.exists():
        return results
    
    for file in rag_path.glob("*_rag_results.json"):
        with open(file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results

def create_embedding_table(results):
    """Create embedding-only comparison table"""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    df['model_short'] = df['model'].apply(lambda x: x.split('/')[-1])
    
    cols = ['model_short', 'recall@1', 'recall@3', 'recall@5', 'recall@10', 
            'avg_query_time_ms', 'total_queries']
    
    df_display = df[cols].copy()
    
    # Round numeric columns
    numeric_cols = ['recall@1', 'recall@3', 'recall@5', 'recall@10', 'avg_query_time_ms']
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(4)
    
    df_display = df_display.rename(columns={
        'model_short': 'Model',
        'recall@1': 'R@1',
        'recall@3': 'R@3',
        'recall@5': 'R@5',
        'recall@10': 'R@10',
        'avg_query_time_ms': 'Avg Time (ms)',
        'total_queries': 'Queries'
    })
    
    return df_display

def create_rag_table(results):
    """Create RAG evaluation comparison table"""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    df['model_short'] = df['embedding_model'].apply(lambda x: x.split('/')[-1])
    
    cols = [
        'model_short',
        'retrieval_recall',
        'avg_bleu',
        'avg_rouge1',
        'avg_rouge2',
        'avg_rougeL',
        'avg_f1',
        'avg_exact_match'
    ]
    
    df_display = df[cols].copy()
    
    # Round numeric columns
    numeric_cols = ['retrieval_recall', 'avg_bleu', 'avg_rouge1', 'avg_rouge2', 
                   'avg_rougeL', 'avg_f1', 'avg_exact_match']
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(4)
    
    df_display = df_display.rename(columns={
        'model_short': 'Model',
        'retrieval_recall': 'Retrieval R@5',
        'avg_bleu': 'BLEU',
        'avg_rouge1': 'ROUGE-1',
        'avg_rouge2': 'ROUGE-2',
        'avg_rougeL': 'ROUGE-L',
        'avg_f1': 'F1',
        'avg_exact_match': 'Exact Match'
    })
    
    return df_display

def create_combined_table(emb_results, rag_results):
    """Create combined view of embedding + RAG metrics"""
    if not emb_results or not rag_results:
        return None
    
    df_emb = pd.DataFrame(emb_results)
    df_rag = pd.DataFrame(rag_results)
    
    # Simplify model names
    df_emb['model_short'] = df_emb['model'].apply(lambda x: x.split('/')[-1])
    df_rag['model_short'] = df_rag['embedding_model'].apply(lambda x: x.split('/')[-1])
    
    # Merge on model name
    df_combined = df_emb.merge(df_rag, on='model_short', suffixes=('_emb', '_rag'))
    
    # Calculate overall score
    # Weighted: 40% ROUGE-L, 30% F1, 20% Retrieval, 10% BLEU
    df_combined['overall_score'] = (
        0.4 * df_combined['avg_rougeL'] +
        0.3 * df_combined['avg_f1'] +
        0.2 * df_combined['retrieval_recall'] +
        0.1 * df_combined['avg_bleu']
    )
    
    # Select display columns
    cols = [
        'model_short',
        'recall@5',
        'avg_query_time_ms',
        'retrieval_recall',
        'avg_bleu',
        'avg_rougeL',
        'avg_f1',
        'avg_exact_match',
        'overall_score'
    ]
    
    df_display = df_combined[cols].copy()
    
    # Round numeric columns
    numeric_cols = [c for c in cols if c != 'model_short']
    for col in numeric_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(4)
    
    # Rename
    df_display = df_display.rename(columns={
        'model_short': 'Model',
        'recall@5': 'Embedding R@5',
        'avg_query_time_ms': 'Speed (ms)',
        'retrieval_recall': 'RAG R@5',
        'avg_bleu': 'BLEU',
        'avg_rougeL': 'ROUGE-L',
        'avg_f1': 'F1',
        'avg_exact_match': 'Exact Match',
        'overall_score': 'Overall Score'
    })
    
    # Sort by overall score
    df_display = df_display.sort_values('Overall Score', ascending=False)
    
    return df_display

def print_embedding_summary(df):
    """Print embedding evaluation summary"""
    if df is None:
        return
    
    print(f"\n{'='*80}")
    print("📊 EMBEDDING EVALUATION SUMMARY (Retrieval Only)")
    print(f"{'='*80}\n")
    
    print(df.to_string(index=False))
    
    if 'R@5' in df.columns:
        best_idx = df['R@5'].idxmax()
        best = df.iloc[best_idx]
        
        print(f"\n🏆 BEST RETRIEVAL:")
        print(f"   Model: {best['Model']}")
        print(f"   Recall@5: {best['R@5']:.4f} ({best['R@5']*100:.2f}%)")
    
    print(f"\n{'='*80}")

def print_rag_summary(df):
    """Print RAG evaluation summary"""
    if df is None:
        return
    
    print(f"\n{'='*80}")
    print("📊 END-TO-END RAG EVALUATION SUMMARY (Retrieval + LLM)")
    print(f"{'='*80}\n")
    
    print(df.to_string(index=False))
    
    if 'ROUGE-L' in df.columns:
        best_idx = df['ROUGE-L'].idxmax()
        best = df.iloc[best_idx]
        
        print(f"\n🏆 BEST RAG PERFORMANCE:")
        print(f"   Model: {best['Model']}")
        print(f"   ROUGE-L: {best['ROUGE-L']:.4f}")
        print(f"   F1: {best['F1']:.4f}")
        print(f"   Retrieval R@5: {best['Retrieval R@5']:.4f} ({best['Retrieval R@5']*100:.2f}%)")
    
    print(f"\n{'='*80}")

def print_combined_summary(df):
    """Print combined summary with recommendations"""
    if df is None:
        return
    
    print(f"\n{'='*80}")
    print("🎯 COMBINED EVALUATION SUMMARY (Complete RAG Pipeline)")
    print(f"{'='*80}\n")
    
    print(df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("🏆 RANKINGS & RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    # Overall winner
    best_overall = df.iloc[0]  # Already sorted by overall score
    print(f"🥇 OVERALL WINNER: {best_overall['Model']}")
    print(f"   Overall Score: {best_overall['Overall Score']:.4f}")
    print(f"   ROUGE-L: {best_overall['ROUGE-L']:.4f}")
    print(f"   F1: {best_overall['F1']:.4f}")
    print(f"   RAG R@5: {best_overall['RAG R@5']:.4f} ({best_overall['RAG R@5']*100:.2f}%)")
    
    # Fastest
    fastest_idx = df['Speed (ms)'].idxmin()
    fastest = df.iloc[fastest_idx]
    print(f"\n⚡ FASTEST: {fastest['Model']}")
    print(f"   Speed: {fastest['Speed (ms)']:.2f} ms")
    print(f"   Overall Score: {fastest['Overall Score']:.4f}")
    
    # Best retrieval
    best_ret_idx = df['RAG R@5'].idxmax()
    best_ret = df.iloc[best_ret_idx]
    print(f"\n📈 BEST RETRIEVAL: {best_ret['Model']}")
    print(f"   RAG R@5: {best_ret['RAG R@5']:.4f} ({best_ret['RAG R@5']*100:.2f}%)")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"\n  Production Use Cases:")
    print(f"    • Best Overall Quality: {best_overall['Model']}")
    print(f"    • Speed-Critical Apps: {fastest['Model']}")
    print(f"    • High-Precision RAG: {best_ret['Model']}")
    
    print(f"\n  Score Interpretation:")
    if best_overall['Overall Score'] >= 0.35:
        print(f"    ✅ Excellent RAG performance (Score ≥ 0.35)")
    elif best_overall['Overall Score'] >= 0.25:
        print(f"    ✅ Good RAG performance (Score ≥ 0.25)")
    else:
        print(f"    ⚠️  Consider improving retrieval or fine-tuning")
    
    print(f"\n{'='*80}")

def save_markdown_report(emb_df, rag_df, combined_df):
    """Save comprehensive markdown report"""
    md_file = f"{RESULTS_DIR}/COMPLETE_EVALUATION_REPORT.md"
    
    with open(md_file, 'w') as f:
        f.write("# Complete RAG System Evaluation Report\n\n")
        f.write(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        f.write("## Executive Summary\n\n")
        if combined_df is not None:
            best = combined_df.iloc[0]
            f.write(f"**Recommended Model**: `{best['Model']}`\n\n")
            f.write(f"- Overall Score: **{best['Overall Score']:.4f}**\n")
            f.write(f"- ROUGE-L: **{best['ROUGE-L']:.4f}**\n")
            f.write(f"- F1: **{best['F1']:.4f}**\n")
            f.write(f"- Retrieval Recall@5: **{best['RAG R@5']:.4f}** ({best['RAG R@5']*100:.1f}%)\n\n")
        
        f.write("---\n\n")
        
        if emb_df is not None:
            f.write("## 1. Embedding-Only Evaluation (Retrieval)\n\n")
            f.write(emb_df.to_markdown(index=False))
            f.write("\n\n**Metrics**:\n")
            f.write("- R@K: Recall at K (% of queries with gold doc in top-K)\n")
            f.write("- Avg Time: Query encoding time in milliseconds\n\n")
            f.write("---\n\n")
        
        if rag_df is not None:
            f.write("## 2. End-to-End RAG Evaluation (Retrieval + LLM)\n\n")
            f.write(rag_df.to_markdown(index=False))
            f.write("\n\n**Metrics**:\n")
            f.write("- BLEU: N-gram overlap (0-1, higher = better)\n")
            f.write("- ROUGE-1/2/L: Unigram/Bigram/LCS overlap\n")
            f.write("- F1: Token-level precision/recall\n")
            f.write("- Exact Match: % of perfect matches\n\n")
            f.write("---\n\n")
        
        if combined_df is not None:
            f.write("## 3. Combined Performance Overview\n\n")
            f.write(combined_df.to_markdown(index=False))
            f.write("\n\n**Overall Score Formula**:\n")
            f.write("```\n")
            f.write("Overall Score = (0.4 × ROUGE-L) + (0.3 × F1) + (0.2 × Retrieval R@5) + (0.1 × BLEU)\n")
            f.write("```\n\n")
            f.write("**Interpretation**:\n")
            f.write("- Score ≥ 0.35: Excellent RAG system\n")
            f.write("- Score ≥ 0.25: Good RAG system\n")
            f.write("- Score < 0.25: Consider improvements\n\n")
        
        f.write("---\n\n")
        f.write("## 4. Recommendations\n\n")
        
        if combined_df is not None:
            fastest = combined_df.loc[combined_df['Speed (ms)'].idxmin()]
            best_ret = combined_df.loc[combined_df['RAG R@5'].idxmax()]
            
            f.write(f"### Production Deployment\n\n")
            f.write(f"| Use Case | Recommended Model | Reason |\n")
            f.write(f"|----------|------------------|--------|\n")
            f.write(f"| General Purpose | `{best['Model']}` | Best overall quality |\n")
            f.write(f"| Speed-Critical | `{fastest['Model']}` | Fastest ({fastest['Speed (ms)']:.1f}ms) |\n")
            f.write(f"| High Precision | `{best_ret['Model']}` | Best retrieval ({best_ret['RAG R@5']*100:.1f}%) |\n\n")
        
        f.write("### Next Steps\n\n")
        f.write("1. ✅ Choose embedding model based on use case\n")
        f.write("2. ✅ Integrate with production RAG pipeline\n")
        f.write("3. ✅ Monitor performance on real user queries\n")
        f.write("4. ✅ A/B test if needed\n")
        f.write("5. ✅ Re-evaluate periodically with new data\n\n")
    
    return md_file

def main():
    print("📊 Loading all evaluation results...\n")
    
    # Load results
    emb_results = load_embedding_results()
    rag_results = load_rag_results()
    
    # Create tables
    emb_df = create_embedding_table(emb_results) if emb_results else None
    rag_df = create_rag_table(rag_results) if rag_results else None
    combined_df = create_combined_table(emb_results, rag_results) if (emb_results and rag_results) else None
    
    # Print summaries
    if emb_df is not None:
        print_embedding_summary(emb_df)
        emb_df.to_csv(f"{RESULTS_DIR}/embedding_comparison.csv", index=False)
        print(f"💾 Embedding results saved: {RESULTS_DIR}/embedding_comparison.csv")
    else:
        print("⚠️  No embedding results found. Run: run_embedding_eval.bat")
    
    if rag_df is not None:
        print_rag_summary(rag_df)
        rag_df.to_csv(f"{RAG_RESULTS_DIR}/rag_comparison.csv", index=False)
        print(f"💾 RAG results saved: {RAG_RESULTS_DIR}/rag_comparison.csv")
    else:
        print("\n⚠️  No RAG results found. Run: run_full_rag_eval.bat")
    
    if combined_df is not None:
        print_combined_summary(combined_df)
        combined_df.to_csv(f"{RESULTS_DIR}/combined_comparison.csv", index=False)
        print(f"💾 Combined results saved: {RESULTS_DIR}/combined_comparison.csv")
    
    # Save comprehensive report
    if emb_df is not None or rag_df is not None:
        md_file = save_markdown_report(emb_df, rag_df, combined_df)
        print(f"\n📄 Complete report saved: {md_file}")
    
    print(f"\n✅ Comparison tables complete!")

if __name__ == "__main__":
    main()
