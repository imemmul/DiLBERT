import requests
import json
import time
import pandas as pd
from typing import List, Dict, Any
import os
from tqdm import tqdm

# Configuration
API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
BATCH_SIZE = 20
MODEL = "sonar-pro"  # Options: "sonar", "sonar-pro", "sonar-deep-research"
CHECKPOINT_FREQUENCY = 5  # Save after every 5 batches

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def create_batch_prompt(texts: List[Dict[str, Any]]) -> str:
    """Create a prompt for batch processing of texts."""
    prompt = """Analyze each of the following Turkish texts for hate speech. For each text:
1. Determine if it contains hate speech (h or nh)
2. If it contains hate speech, identify the target (religion (r), ethnicity (e), politics (p))

Respond with a JSON array where each object has the following structure:
{
  "id": [text_id],
  "label": [h/nh],
  "target": [target category (r, e, p) or null if not hate speech]
}

Here are the texts to analyze:

"""
    for text_item in texts:
        prompt += f"ID: {text_item['id']}\nText: {text_item['text']}\n\n"
    
    return prompt

def call_perplexity_api(prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """Call the Perplexity API with retry logic."""
    url = "https://api.perplexity.ai/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"API call failed after {max_retries} attempts: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff

def extract_json_from_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and parse JSON from the API response with better error handling."""
    content = response['choices'][0]['message']['content']
    
    # Try to extract a JSON array first (most common case)
    json_start = content.find('[')
    json_end = content.rfind(']') + 1
    
    if json_start >= 0 and json_end > json_start:
        try:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If parsing fails, continue to alternative methods
            pass
    
    # If array parsing fails, try to find individual JSON objects
    import re
    results = []
    
    # Method 1: Look for complete JSON objects with our expected fields
    pattern = r'{\s*"id"\s*:\s*(\d+)\s*,\s*"label"\s*:\s*"([^"]+)"\s*,\s*"target"\s*:\s*([^,}]+)\s*}'
    matches = re.findall(pattern, content)
    
    if matches:
        for match in matches:
            id_val, label, target = match
            # Handle null/None values properly
            if target.strip().lower() in ('null', 'none'):
                target_val = None
            else:
                # Remove quotes if present
                target_val = target.strip('"\'')
            
            results.append({
                "id": int(id_val),
                "label": label,
                "target": target_val
            })
        return results
    
    # Method 2: Try to extract each JSON object separately
    object_matches = re.finditer(r'{[^{]*?}', content)
    for match in object_matches:
        try:
            obj = json.loads(match.group(0))
            if "id" in obj and "label" in obj:  # Verify it's one of our objects
                results.append(obj)
        except json.JSONDecodeError:
            continue
    
    if results:
        return results
        
    # If we still haven't found anything, show the content and raise an error
    error_msg = f"Could not extract valid JSON. Response content excerpt: {content[:200]}..."
    raise ValueError(error_msg)

def process_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process a batch of texts and return annotations."""
    prompt = create_batch_prompt(batch)
    response = call_perplexity_api(prompt)
    results = extract_json_from_response(response)
    return results

def save_checkpoint(results: List[Dict[str, Any]], output_path: str, checkpoint_suffix: str = "_checkpoint"):
    """Save current results to a checkpoint file."""
    checkpoint_path = output_path.replace('.csv', f'{checkpoint_suffix}.csv')
    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(checkpoint_path: str) -> List[Dict[str, Any]]:
    """Load results from a checkpoint file if it exists."""
    if os.path.exists(checkpoint_path):
        df = pd.read_csv(checkpoint_path)
        # Convert to list of dictionaries
        results = df.to_dict('records')
        print(f"Loaded {len(results)} annotations from checkpoint")
        return results
    return []

def get_processed_ids(results: List[Dict[str, Any]]) -> set:
    """Get the set of IDs that have already been processed."""
    return {item['id'] for item in results}

def annotate_dataset(file_path: str, output_path: str, resume: bool = True):
    """Annotate all texts in a dataset file with frequent checkpoints."""
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Add an ID column if it doesn't exist
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)
    
    # Ensure there's a text column
    if 'text' not in df.columns:
        # Try to find the text column - it might have a different name
        text_column_candidates = ['text', 'content', 'tweet', 'message', 'comment']
        found = False
        for column in text_column_candidates:
            if column in df.columns:
                df['text'] = df[column]
                found = True
                break
        
        if not found:
            # If no obvious text column, use the first string column
            for column in df.columns:
                if df[column].dtype == 'object':  # String columns are typically 'object' dtype
                    df['text'] = df[column]
                    print(f"Using column '{column}' as the text column")
                    found = True
                    break
        
        if not found:
            raise ValueError("Could not identify a text column in the dataset")
    
    # Check for checkpoint
    checkpoint_path = output_path.replace('.csv', '_checkpoint.csv')
    all_results = []
    
    if resume and os.path.exists(checkpoint_path):
        all_results = load_checkpoint(checkpoint_path)
        processed_ids = get_processed_ids(all_results)
        print(f"Resuming from checkpoint. {len(processed_ids)} texts already processed.")
    else:
        processed_ids = set()
        print("Starting new annotation process.")
    
    # Prepare data structure, excluding already processed texts
    texts = [{"id": row['id'], "text": row['text']} 
             for _, row in df.iterrows() 
             if row['id'] not in processed_ids]
    
    if not texts:
        print("All texts have already been processed. Nothing to do.")
        # Save final results
        pd.DataFrame(all_results).to_csv(output_path, index=False)
        return pd.DataFrame(all_results)
    
    batches = chunk_list(texts, BATCH_SIZE)
    
    # Process all batches with progress bar
    for idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        try:
            results = process_batch(batch)
            all_results.extend(results)
            
            # Save checkpoint periodically
            if (idx + 1) % CHECKPOINT_FREQUENCY == 0 or idx == len(batches) - 1:
                save_checkpoint(all_results, output_path)
                
            if idx % 20 == 0:
                print(f"Processed {idx * BATCH_SIZE} texts...")
                print(f"results: {results}")
                
        except Exception as e:
            print(f"Error processing batch {idx}: {str(e)}")
            # Save what we have so far
            save_checkpoint(all_results, output_path, f"_error_batch{idx}")
            print("Saved results up to the error point.")
            raise
    
    # Create final results dataframe
    results_df = pd.DataFrame(all_results)
    
    # Save final results
    results_df.to_csv(output_path, index=False)
    
    # Remove checkpoint file if everything completed successfully
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            print(f"Removed checkpoint file as process completed successfully")
        except Exception as e:
            print(f"Note: Could not remove checkpoint file: {str(e)}")
    
    print(f"\nAnnotation Complete!")
    print(f"Total texts processed: {len(texts)}")
    print(f"Total annotations: {len(all_results)}")
    
    return results_df

if __name__ == "__main__":
    input_file = "/Users/emirulurak/Desktop/dev/ozu/cs549/DiLBERT/turkish_reddit_hate_speech_dataset.csv"
    output_file = "./annotated_hate_speech.csv"
    
    # Set resume=True to continue from checkpoint if it exists
    annotate_dataset(input_file, output_file, resume=True)
