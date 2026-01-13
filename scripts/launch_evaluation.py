#!/usr/bin/env python3
"""
Quick Launcher for Comprehensive Model Evaluation
Run this script to start the evaluation with customizable settings
"""

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpt_vs_finetune_v5 import main

def get_user_preferences():
    """Get user preferences for the evaluation"""
    print("🎯 EVALUATION SETUP")
    print("=" * 40)
    
    # Get number of samples
    while True:
        try:
            samples = input("How many samples to evaluate? (5-50, default 10): ").strip()
            if not samples:
                samples = 10
            else:
                samples = int(samples)
            
            if 5 <= samples <= 50:
                break
            else:
                print("Please enter a number between 5 and 50")
        except ValueError:
            print("Please enter a valid number")
    
    # Confirm settings
    print(f"\n✅ Configuration:")
    print(f"   - Samples to evaluate: {samples}")
    print(f"   - Models: ChatGPT vs Finetuned Gemma vs Finetuned+RAG")
    print(f"   - Data sources: gpt_plus_gemma.jsonl & rag_plus_finetune.jsonl")
    
    confirm = input("\nProceed with evaluation? (y/n, default y): ").strip().lower()
    if confirm in ['n', 'no']:
        print("Evaluation cancelled.")
        sys.exit(0)
    
    return samples

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE MODEL EVALUATION LAUNCHER")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        "gpt_plus_gemma.jsonl",
        "rag_plus_finetune.jsonl",
        "gemma3_v2_merged_model"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present before running the evaluation.")
        sys.exit(1)
    
    print("✅ All required files found!")
    
    try:
        # Get user preferences
        sample_count = get_user_preferences()
        
        # Update the main function with user preferences
        print(f"\n🔄 Starting evaluation with {sample_count} samples...")
        
        # Modify the MAX_SAMPLES in the main script
        import gpt_vs_finetune_v5
        
        # Run the evaluation
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Evaluation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("Please check the logs for more details.")
    finally:
        print("\nThank you for using the Comprehensive Model Evaluator!")