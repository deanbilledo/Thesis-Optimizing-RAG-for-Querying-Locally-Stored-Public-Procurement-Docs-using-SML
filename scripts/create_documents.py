#!/usr/bin/env python3
"""
Create documents.jsonl from train-weska-clean.jsonl
Chunks answers into ~100-300 word passages and assigns document IDs
"""

import json
import re

# Configuration
INPUT_FILE = "train-weska-clean.jsonl"
OUTPUT_FILE = "documents.jsonl"
MIN_CHUNK_WORDS = 50   # Minimum words per chunk
MAX_CHUNK_WORDS = 300  # Maximum words per chunk

def split_into_sentences(text):
    """Split text into sentences"""
    # Split on periods, question marks, exclamation marks followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_text(text, min_words=MIN_CHUNK_WORDS, max_words=MAX_CHUNK_WORDS):
    """
    Chunk text into passages of ~100-300 words
    Tries to preserve sentence boundaries
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed max, save current chunk
        if current_word_count + sentence_words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words
            
            # If we've reached a good chunk size, save it
            if current_word_count >= min_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_word_count = 0
    
    # Add remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def main():
    print(f"📂 Reading from: {INPUT_FILE}")
    
    # Load all data
    data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"📊 Total samples: {len(data)}")
    
    # Create documents from outputs
    documents = []
    doc_id_counter = 1
    
    for item in data:
        # Use the output (answer) as the source text
        text = item["output"]
        
        # Option 1: Use full answer as one document (simpler)
        # Uncomment this if you want one document per answer
        # documents.append({
        #     "id": f"doc_{doc_id_counter}",
        #     "text": text,
        #     "source_instruction": item["instruction"]  # Optional: keep track of original question
        # })
        # doc_id_counter += 1
        
        # Option 2: Chunk longer answers into smaller passages (better for retrieval)
        chunks = chunk_text(text)
        
        for chunk in chunks:
            # Only add chunks with reasonable length
            word_count = len(chunk.split())
            if word_count >= 20:  # Skip very short chunks
                documents.append({
                    "id": f"doc_{doc_id_counter}",
                    "text": chunk,
                    "source_instruction": item["instruction"]  # Optional metadata
                })
                doc_id_counter += 1
    
    print(f"📚 Created {len(documents)} document chunks")
    
    # Write to output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    print(f"✅ Saved documents to: {OUTPUT_FILE}")
    
    # Statistics
    word_counts = [len(doc["text"].split()) for doc in documents]
    avg_words = sum(word_counts) / len(word_counts)
    print(f"\n📊 Statistics:")
    print(f"   Total documents: {len(documents)}")
    print(f"   Average words per document: {avg_words:.1f}")
    print(f"   Min words: {min(word_counts)}")
    print(f"   Max words: {max(word_counts)}")
    
    # Print samples
    print("\n📋 Sample documents:")
    for i in range(min(3, len(documents))):
        print(f"\n--- Document {documents[i]['id']} ---")
        print(f"Text: {documents[i]['text'][:150]}...")
        print(f"Words: {len(documents[i]['text'].split())}")

if __name__ == "__main__":
    main()
