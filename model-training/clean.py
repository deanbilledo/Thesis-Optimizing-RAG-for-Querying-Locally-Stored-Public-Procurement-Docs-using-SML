import re

def clean_dataset_regex(input_file, output_file):
    """
    Remove sequences of 3 or more newlines using regex
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Remove 3 or more consecutive newlines, replace with double newline
        cleaned_content = re.sub(r'\n{3,}', '\n\n', content)
        
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
        
        print(f"Successfully cleaned dataset. Output saved to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")

# Usage
if __name__ == "__main__":
    input_filename = "key-responses.jsonl"
    output_filename = "key-responses-cleaned.jsonl"
    
    clean_dataset_regex(input_filename, output_filename)