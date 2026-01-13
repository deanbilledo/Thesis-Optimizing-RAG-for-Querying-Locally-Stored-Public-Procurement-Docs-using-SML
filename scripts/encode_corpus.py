#!/usr/bin/env python3
"""
Encode document corpus using sentence embedding models
Creates embeddings for all documents for retrieval evaluation
"""

from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import argparse
import os

def encode_corpus(model_name, document_path, output_path):
    """
    Encode all documents using specified embedding model
    
    Args:
        model_name: HuggingFace model name
        document_path: Path to documents.jsonl
        output_path: Path to save embeddings JSON
    """
    print(f"🔄 Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print(f"📂 Reading documents from: {document_path}")
    docs = []
    with open(document_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                docs.append((data['id'], data['text']))
    
    print(f"📊 Total documents: {len(docs)}")

    print(f"🧠 Encoding documents...")
    embeddings = []
    for doc_id, text in tqdm(docs, desc="Encoding"):
        emb = model.encode(text, convert_to_tensor=False).tolist()
        embeddings.append({
            "id": doc_id,
            "embedding": emb
        })

    print(f"💾 Saving embeddings to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(embeddings, f)
    
    print(f"✅ Encoded {len(embeddings)} documents")
    print(f"   Embedding dimension: {len(embeddings[0]['embedding'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode document corpus with embedding model"
    )
    parser.add_argument(
        "--model", 
        required=True,
        help="HuggingFace model name (e.g., all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--docs", 
        default="documents.jsonl",
        help="Path to documents.jsonl"
    )
    parser.add_argument(
        "--output", 
        required=True,
        help="Output path for embeddings JSON"
    )
    args = parser.parse_args()

    encode_corpus(args.model, args.docs, args.output)
    print("\n✅ Encoding complete!")
