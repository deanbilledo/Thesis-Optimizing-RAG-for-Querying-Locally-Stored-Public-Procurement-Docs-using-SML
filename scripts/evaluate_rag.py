#!/usr/bin/env python3
"""
End-to-end RAG evaluation: Retrieval + LLM Answer Generation + Evaluation
"""

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import numpy as np
from tqdm import tqdm
import argparse
import torch
from collections import defaultdict

# Metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except:
    ROUGE_AVAILABLE = False


def load_embeddings(path):
    """Load document embeddings"""
    with open(path, 'r') as f:
        data = json.load(f)
    return {d['id']: np.array(d['embedding']) for d in data}


def load_documents(path):
    """Load original documents"""
    docs = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                docs[data['id']] = data['text']
    return docs


def load_qa_pairs(path):
    """Load Q&A pairs"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieve_top_k(query, embedding_model, doc_embeddings, doc_ids, k=5):
    """
    Retrieve top-K documents for a query
    
    Returns:
        List of (doc_id, score) tuples
    """
    query_emb = embedding_model.encode(query)
    doc_matrix = np.vstack([doc_embeddings[doc_id] for doc_id in doc_ids])
    scores = np.dot(doc_matrix, query_emb)
    top_k_indices = np.argsort(scores)[::-1][:k]
    
    return [(doc_ids[i], scores[i]) for i in top_k_indices]


def create_rag_prompt(query, retrieved_docs, system_prompt, tokenizer):
    """
    Create RAG prompt with retrieved context using official chat template
    """
    context = "\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(retrieved_docs)])
    
    user_message = f"""Context:
{context}

Question: {query}

Please answer the question based on the context provided above."""
    
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # Use official chat template
    prompt = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )
    
    return prompt


def generate_answer(model, tokenizer, prompt, max_new_tokens=150):
    """Generate answer using LLM"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # For quantized models with device_map="auto", inputs are already on correct device
    # Only move to device if model has a single device
    if hasattr(model, 'device'):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    elif hasattr(model, 'hf_device_map'):
        # For models with device_map, use the device of the first layer
        device = list(model.hf_device_map.values())[0]
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only new tokens
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return generated_text.strip()


def calculate_metrics(reference, generated):
    """Calculate BLEU, ROUGE, F1, Exact Match"""
    metrics = {}
    
    # BLEU
    if NLTK_AVAILABLE:
        smoothie = SmoothingFunction().method4
        metrics['bleu'] = sentence_bleu([reference.split()], generated.split(), smoothing_function=smoothie)
    else:
        metrics['bleu'] = 0.0
    
    # ROUGE
    if ROUGE_AVAILABLE:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, generated)
        metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
        metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
        metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
    else:
        metrics['rouge1'] = 0.0
        metrics['rouge2'] = 0.0
        metrics['rougeL'] = 0.0
    
    # F1
    ref_tokens = set(reference.lower().split())
    gen_tokens = set(generated.lower().split())
    if len(ref_tokens) == 0 or len(gen_tokens) == 0:
        metrics['f1'] = 0.0
    else:
        intersection = ref_tokens & gen_tokens
        if len(intersection) == 0:
            metrics['f1'] = 0.0
        else:
            precision = len(intersection) / len(gen_tokens)
            recall = len(intersection) / len(ref_tokens)
            metrics['f1'] = 2 * precision * recall / (precision + recall)
    
    # Exact Match
    metrics['exact_match'] = 1.0 if reference.strip().lower() == generated.strip().lower() else 0.0
    
    return metrics


def evaluate_rag(
    embedding_model_name,
    embeddings_file,
    documents_file,
    qa_file,
    llm_base_model,
    llm_adapter=None,
    top_k=5,
    max_new_tokens=150,
    system_prompt="You are a helpful assistant specialized in Philippine government procurement."
):
    """
    End-to-end RAG evaluation
    """
    print(f"\n{'='*70}")
    print(f"END-TO-END RAG EVALUATION")
    print(f"{'='*70}\n")
    
    # Load embedding model
    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # Load embeddings and documents
    doc_embeddings = load_embeddings(embeddings_file)
    documents = load_documents(documents_file)
    doc_ids = list(doc_embeddings.keys())
    
    # Load Q&A pairs
    qa_data = load_qa_pairs(qa_file)
    
    # Load LLM - Using FP16 instead of 4-bit to avoid device dispatch issues
    print(f"Loading LLM: {llm_base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(llm_base_model, clean_up_tokenization_spaces=True)
    
    # Use FP16 on GPU instead of 4-bit quantization for compatibility
    if torch.cuda.is_available():
        base_llm = AutoModelForCausalLM.from_pretrained(
            llm_base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        base_llm = AutoModelForCausalLM.from_pretrained(
            llm_base_model,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    if llm_adapter:
        print(f"Loading adapter: {llm_adapter}")
        llm = PeftModel.from_pretrained(base_llm, llm_adapter)
    else:
        llm = base_llm
    
    print(f"\nEvaluating {len(qa_data)} queries with RAG...")
    
    # Metrics accumulator
    all_metrics = defaultdict(list)
    retrieval_hits = 0
    
    for item in tqdm(qa_data, desc="RAG Evaluation"):
        query = item['query']
        reference_answer = item['answer']
        gold_doc_id = item['gold_document_id']
        
        # Retrieve top-K documents
        top_docs = retrieve_top_k(query, embedding_model, doc_embeddings, doc_ids, k=top_k)
        retrieved_doc_ids = [doc_id for doc_id, _ in top_docs]
        retrieved_texts = [documents[doc_id] for doc_id in retrieved_doc_ids]
        
        # Check if gold document was retrieved
        if gold_doc_id in retrieved_doc_ids:
            retrieval_hits += 1
        
        # Create RAG prompt
        prompt = create_rag_prompt(query, retrieved_texts, system_prompt, tokenizer)
        
        # Generate answer
        generated_answer = generate_answer(llm, tokenizer, prompt, max_new_tokens)
        
        # Calculate metrics
        metrics = calculate_metrics(reference_answer, generated_answer)
        for k, v in metrics.items():
            all_metrics[k].append(v)
    
    # Aggregate results
    results = {
        "embedding_model": embedding_model_name,
        "llm_model": llm_base_model,
        "llm_adapter": llm_adapter if llm_adapter else "None",
        "top_k": top_k,
        "total_queries": len(qa_data),
        "retrieval_recall": retrieval_hits / len(qa_data),
    }
    
    for metric_name, values in all_metrics.items():
        results[f"avg_{metric_name}"] = np.mean(values)
    
    return results


def print_results(results):
    """Pretty print results"""
    print(f"\n{'='*70}")
    print(f"RAG EVALUATION RESULTS")
    print(f"{'='*70}\n")
    
    print(f"Embedding Model: {results['embedding_model']}")
    print(f"LLM Model: {results['llm_model']}")
    print(f"LLM Adapter: {results['llm_adapter']}")
    print(f"Top-K Retrieved: {results['top_k']}")
    print(f"Total Queries: {results['total_queries']}")
    
    print(f"\nRetrieval Performance:")
    print(f"  Recall@{results['top_k']}: {results['retrieval_recall']:.4f} ({results['retrieval_recall']*100:.2f}%)")
    
    print(f"\nGeneration Quality:")
    print(f"  BLEU: {results['avg_bleu']:.4f}")
    print(f"  ROUGE-1: {results['avg_rouge1']:.4f}")
    print(f"  ROUGE-2: {results['avg_rouge2']:.4f}")
    print(f"  ROUGE-L: {results['avg_rougeL']:.4f}")
    print(f"  F1: {results['avg_f1']:.4f}")
    print(f"  Exact Match: {results['avg_exact_match']:.4f}")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end RAG evaluation")
    parser.add_argument("--embedding-model", required=True, help="Embedding model name")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings JSON")
    parser.add_argument("--documents", default="documents.jsonl", help="Path to documents.jsonl")
    parser.add_argument("--qa", default="qa_pairs.jsonl", help="Path to qa_pairs.jsonl")
    parser.add_argument("--llm", required=True, help="Base LLM model")
    parser.add_argument("--adapter", default=None, help="Optional: LLM adapter path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--max-tokens", type=int, default=150, help="Max new tokens for generation")
    parser.add_argument("--output", default="rag_results.json", help="Output file for results")
    args = parser.parse_args()
    
    results = evaluate_rag(
        embedding_model_name=args.embedding_model,
        embeddings_file=args.embeddings,
        documents_file=args.documents,
        qa_file=args.qa,
        llm_base_model=args.llm,
        llm_adapter=args.adapter,
        top_k=args.top_k,
        max_new_tokens=args.max_tokens
    )
    
    print_results(results)
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
