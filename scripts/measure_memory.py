"""
Memory Usage Measurement Script
===============================
Measures actual RAM usage of the RAG system components.
"""

import os
import sys
import gc
import psutil
from pathlib import Path

# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch


def get_memory_mb():
    """Get current process memory in MB"""
    return psutil.Process().memory_info().rss / (1024 ** 2)


def main():
    workspace = Path(__file__).parent.parent
    
    print("=" * 60)
    print("  RAG SYSTEM MEMORY USAGE MEASUREMENT")
    print("=" * 60)
    
    # System baseline
    gc.collect()
    baseline = get_memory_mb()
    print(f"\n[Baseline] Python process: {baseline:.1f} MB")
    
    # 1. Load embedding model
    print("\n[1] Loading Embedding Model...")
    from sentence_transformers import SentenceTransformer
    
    embed_path = workspace / 'embedding_model'
    embed_model = SentenceTransformer(str(embed_path), device='cpu')
    
    after_embed = get_memory_mb()
    embed_usage = after_embed - baseline
    print(f"    Memory after embedding model: {after_embed:.1f} MB")
    print(f"    Embedding model usage: {embed_usage:.1f} MB")
    
    # 2. Load LLM
    print("\n[2] Loading LLM (Gemma-3-1B + LoRA)...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    base_model_path = workspace / 'base_model'
    model_path = workspace / 'model'
    
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path))
    
    after_tokenizer = get_memory_mb()
    tokenizer_usage = after_tokenizer - after_embed
    print(f"    Memory after tokenizer: {after_tokenizer:.1f} MB (+{tokenizer_usage:.1f} MB)")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True
    )
    
    after_base = get_memory_mb()
    base_usage = after_base - after_tokenizer
    print(f"    Memory after base model: {after_base:.1f} MB (+{base_usage:.1f} MB)")
    
    llm = PeftModel.from_pretrained(base_model, str(model_path))
    llm = llm.to('cpu')
    llm.eval()
    
    after_lora = get_memory_mb()
    lora_usage = after_lora - after_base
    print(f"    Memory after LoRA adapter: {after_lora:.1f} MB (+{lora_usage:.1f} MB)")
    
    # 3. Load ChromaDB
    print("\n[3] Loading ChromaDB...")
    import chromadb
    from chromadb.config import Settings
    
    client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
    collection = client.create_collection(name="test", metadata={"hnsw:space": "cosine"})
    
    after_chroma = get_memory_mb()
    chroma_usage = after_chroma - after_lora
    print(f"    Memory after ChromaDB: {after_chroma:.1f} MB (+{chroma_usage:.1f} MB)")
    
    # 4. Add some test documents
    print("\n[4] Adding 100 test documents...")
    test_docs = [f"Test procurement document {i} with budget PHP {i*1000:,.2f}" for i in range(100)]
    embeddings = embed_model.encode(test_docs)
    collection.add(
        documents=test_docs,
        embeddings=embeddings.tolist(),
        ids=[f"doc_{i}" for i in range(100)]
    )
    
    after_docs = get_memory_mb()
    docs_usage = after_docs - after_chroma
    print(f"    Memory after 100 docs: {after_docs:.1f} MB (+{docs_usage:.1f} MB)")
    
    # Summary
    total_usage = after_docs - baseline
    
    print("\n" + "=" * 60)
    print("  MEMORY USAGE SUMMARY")
    print("=" * 60)
    print(f"""
| Component              | Memory Usage |
|------------------------|--------------|
| Baseline (Python)      | {baseline:.1f} MB |
| Embedding Model        | {embed_usage:.1f} MB |
| Tokenizer              | {tokenizer_usage:.1f} MB |
| Gemma-3-1B Base        | {base_usage:.1f} MB |
| LoRA Adapter           | {lora_usage:.1f} MB |
| ChromaDB               | {chroma_usage:.1f} MB |
| 100 Documents          | {docs_usage:.1f} MB |
|------------------------|--------------|
| **TOTAL RAG SYSTEM**   | **{total_usage:.1f} MB** |
| **In GB**              | **{total_usage/1024:.2f} GB** |
""")
    
    print("=" * 60)
    print(f"  Minimum RAM Required: ~{(total_usage/1024 + 2):.0f} GB")
    print(f"  (includes ~2GB for OS and background processes)")
    print("=" * 60)


if __name__ == "__main__":
    main()
