# llama_dataset_optimizer/core/deduplicator.py

import torch
import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from ..utils.llama_utils import apply_chat_template

def get_llama_embeddings(dataset, config, tokenizer, batch_size=32):
    """
    Generates semantic embeddings for each sample in the dataset using a Llama model.
    """
    model_name = config['deduplication']['embedding_model']
    print(f"Generating embeddings using model: {model_name}")
    
    # We use AutoModel for embeddings, not AutoModelForCausalLM
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating Embeddings"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        
        texts = [apply_chat_template(sample, tokenizer) for sample in batch]
        
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Mean pool the last hidden state
            last_hidden_states = outputs.hidden_states[-1]
            # [batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim]
            batch_embeddings = last_hidden_states.mean(dim=1)

        all_embeddings.append(batch_embeddings.cpu().numpy())
        
    return np.vstack(all_embeddings)


def deduplicate_faiss_gpu(embeddings: np.ndarray, threshold: float):
    """
    Finds and removes near-duplicates from a set of embeddings using FAISS on GPU.
    Returns the indices of the items to keep.
    """
    print(f"Deduplicating {len(embeddings)} samples with threshold {threshold}...")
    
    if not torch.cuda.is_available():
        raise SystemError("FAISS-GPU deduplication requires a CUDA-enabled GPU.")
        
    d = embeddings.shape[1]
    embeddings = embeddings.astype('float32') # FAISS requires float32
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build a FAISS index
    index = faiss.IndexFlatIP(d) # Using Inner Product (IP) for cosine similarity on normalized vectors
    
    # Move the index to all available GPUs
    gpu_index = faiss.index_cpu_to_all_gpus(index)
    gpu_index.add(embeddings)
    
    # Search for nearest neighbors (k=2 to find oneself and the closest neighbor)
    print("Searching for duplicates...")
    distances, indices = gpu_index.search(embeddings, k=2)
    
    # Identify duplicates
    # A sample is a duplicate if its closest neighbor (not itself) has a similarity > threshold
    is_duplicate = distances[:, 1] >= threshold
    
    # To avoid removing both items in a duplicate pair, we only remove the second one.
    # We build a set of items to remove.
    to_remove = set()
    for i, is_dup in enumerate(tqdm(is_duplicate, desc="Identifying duplicates to remove")):
        if is_dup:
            # i is the query, indices[i, 1] is its nearest neighbor
            dup_pair = tuple(sorted((i, indices[i, 1])))
            # Add the second item in the sorted pair to the removal set
            # This ensures we always keep one of them (the one with the smaller index)
            if dup_pair[1] not in to_remove:
                to_remove.add(dup_pair[1])

    all_indices = set(range(len(embeddings)))
    keep_indices = sorted(list(all_indices - to_remove))
    
    print(f"Deduplication complete. Found and removed {len(to_remove)} duplicates. "
          f"Keeping {len(keep_indices)} unique samples.")
    
    return keep_indices