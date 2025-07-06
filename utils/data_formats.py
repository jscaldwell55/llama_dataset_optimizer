# llama_dataset_optimizer/utils/data_formats.py

import json
from datasets import load_dataset, Dataset
from tqdm import tqdm

def load_and_normalize_dataset(path: str, split: str = "train") -> Dataset:
    """
    Loads a dataset from a Hugging Face path or local JSONL file
    and normalizes it into the Llama conversation format.

    The normalized format for each sample is a dictionary:
    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    print(f"Loading and normalizing dataset from: {path}")

    if path.endswith(".jsonl"):
        data = []
        with open(path, "r") as f:
            for line in tqdm(f, desc="Loading JSONL"):
                data.append(json.loads(line))
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset(path, split=split)

    # --- Normalization Logic ---
    # This needs to be adapted based on common dataset structures.
    # We will try to detect the format and convert it.
    
    normalized_data = []
    
    # Example for Ultrachat format: dataset['messages']
    if 'messages' in dataset.column_names and isinstance(dataset[0]['messages'], list):
         print("Detected 'messages' column. Assuming Ultrachat/ShareGPT format.")
         # Already in the correct format
         return dataset

    # Example for Alpaca format: 'instruction', 'input', 'output' or 'response'
    elif 'instruction' in dataset.column_names and ('output' in dataset.column_names or 'response' in dataset.column_names):
        output_key = 'output' if 'output' in dataset.column_names else 'response'
        print(f"Detected 'instruction'/'{output_key}' columns. Converting to Llama format.")
        for sample in tqdm(dataset, desc="Normalizing Alpaca format"):
            user_content = sample['instruction']
            if 'input' in sample and sample['input']:
                user_content += "\n" + sample['input']
            
            normalized_data.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": sample[output_key]}
                ]
            })
        return Dataset.from_list(normalized_data)
        
    else:
        raise ValueError(
            "Unsupported dataset format. Please provide a dataset in either "
            "Ultrachat/ShareGPT format (with a 'messages' column) or "
            "Alpaca format (with 'instruction' and 'output'/'response' columns)."
        )