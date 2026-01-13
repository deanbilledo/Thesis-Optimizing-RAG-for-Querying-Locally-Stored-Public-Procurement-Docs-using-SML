import json

def extract_valid_entries():
    """
    Extract 100 entries from true-data.jsonl and create two files:
    1. rag_plus_finetune.jsonl - entries with instruction, input, and output
    2. gpt_plus_gemma.jsonl - entries with only instruction and output
    """
    
    valid_entries_with_input = []
    valid_entries_without_input = []
    total_processed = 0
    
    print("Processing true-data.jsonl...")
    
    with open('true-data.jsonl', 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    item = json.loads(line.strip())
                    total_processed += 1
                    
                    # Check if all required fields exist and have content
                    if all(key in item for key in ['instruction', 'input', 'output']):
                        if (item['instruction'].strip() and 
                            item['input'].strip() and 
                            item['output'].strip()):
                            
                            # For rag_plus_finetune.jsonl - keep all fields
                            valid_entries_with_input.append(item)
                            
                            # For gpt_plus_gemma.jsonl - only instruction and output
                            entry_without_input = {
                                'instruction': item['instruction'],
                                'output': item['output']
                            }
                            valid_entries_without_input.append(entry_without_input)
                            
                            # Stop when we have 100 valid entries
                            if len(valid_entries_with_input) >= 100:
                                break
                                
                except json.JSONDecodeError:
                    print(f'Error parsing line {line_num}')
    
    print(f"Total lines processed: {total_processed}")
    print(f"Valid entries found: {len(valid_entries_with_input)}")
    
    # Save first file: rag_plus_finetune.jsonl (with input field)
    with open('rag_plus_finetune.jsonl', 'w', encoding='utf-8') as f:
        for entry in valid_entries_with_input:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Created 'rag_plus_finetune.jsonl' with {len(valid_entries_with_input)} entries (instruction + input + output)")
    
    # Save second file: gpt_plus_gemma.jsonl (without input field)
    with open('gpt_plus_gemma.jsonl', 'w', encoding='utf-8') as f:
        for entry in valid_entries_without_input:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Created 'gpt_plus_gemma.jsonl' with {len(valid_entries_without_input)} entries (instruction + output only)")
    
    # Show sample of first 3 entries
    print(f"\nSample from rag_plus_finetune.jsonl (with input):")
    for i, entry in enumerate(valid_entries_with_input[:3]):
        print(f"\n{i+1}. Instruction: {entry['instruction'][:60]}...")
        print(f"   Input: {entry['input'][:60]}...")
        print(f"   Output: {entry['output'][:60]}...")
    
    print(f"\nSample from gpt_plus_gemma.jsonl (without input):")
    for i, entry in enumerate(valid_entries_without_input[:3]):
        print(f"\n{i+1}. Instruction: {entry['instruction'][:60]}...")
        print(f"   Output: {entry['output'][:60]}...")

if __name__ == "__main__":
    extract_valid_entries()
