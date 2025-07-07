# llama_dataset_optimizer/core/deduplicator.py

import torch
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from utils.llama_utils import apply_chat_template

def get_sentence_transformer_embeddings(dataset, config, tokenizer, batch_size=32):
    """
    Generates semantic embeddings using a sentence transformer model for better quality.
    """
    model_name = config['deduplication']['embedding_model']
    print(f"Generating embeddings using sentence transformer: {model_name}")
    
    # Use sentence transformer for better semantic embeddings
    model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract text from dataset samples
    texts = []
    for sample in tqdm(dataset, desc="Processing samples"):
        # Convert conversation to text representation
        text = apply_chat_template(sample, tokenizer)
        texts.append(text)
    
    # Generate embeddings in batches
    all_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # Pre-normalize for cosine similarity
    )
    
    return all_embeddings


def get_llama_embeddings(dataset, config, tokenizer, batch_size=32):
    """
    Generates semantic embeddings for each sample in the dataset using a Llama model.
    Fallback method if sentence transformers are not available.
    """
    model_name = config['deduplication']['embedding_model']
    print(f"Generating embeddings using Llama model: {model_name}")
    
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
        
    embeddings = np.vstack(all_embeddings)
    
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return embeddings


def get_embeddings(dataset, config, tokenizer, batch_size=32):
    """
    Main embedding function that chooses the best available method.
    """
    # Handle empty dataset
    if len(dataset) == 0:
        print("No samples to generate embeddings for.")
        return np.array([])
    
    embedding_method = config['deduplication'].get('method', 'sentence_transformer')
    
    if embedding_method == 'sentence_transformer':
        try:
            return get_sentence_transformer_embeddings(dataset, config, tokenizer, batch_size)
        except Exception as e:
            print(f"Sentence transformer failed: {e}")
            print("Falling back to Llama embeddings...")
            return get_llama_embeddings(dataset, config, tokenizer, batch_size)
    else:
        return get_llama_embeddings(dataset, config, tokenizer, batch_size)


def deduplicate_faiss_gpu(embeddings: np.ndarray, threshold: float):
    """
    Finds and removes near-duplicates from a set of embeddings using FAISS on GPU.
    Returns the indices of the items to keep.
    """
    # Handle empty embeddings
    if len(embeddings) == 0:
        print("No embeddings to deduplicate.")
        return []
    
    print(f"Deduplicating {len(embeddings)} samples with threshold {threshold}...")
    
    d = embeddings.shape[1]
    embeddings = embeddings.astype('float32')  # FAISS requires float32
    
    # Normalize embeddings for cosine similarity if not already normalized
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6):
        print("Normalizing embeddings for cosine similarity...")
        faiss.normalize_L2(embeddings)
    
    # Try GPU first, fall back to CPU if GPU not available
    try:
        if torch.cuda.is_available() and faiss.get_num_gpus() > 0:
            print("Using GPU for FAISS deduplication...")
            index = faiss.IndexFlatIP(d)  # Inner Product for cosine similarity on normalized vectors
            gpu_index = faiss.index_cpu_to_all_gpus(index)
            gpu_index.add(embeddings)
            search_index = gpu_index
        else:
            print("GPU not available, using CPU for FAISS deduplication...")
            search_index = faiss.IndexFlatIP(d)
            search_index.add(embeddings)
    except Exception as e:
        print(f"FAISS GPU setup failed: {e}")
        print("Falling back to CPU...")
        search_index = faiss.IndexFlatIP(d)
        search_index.add(embeddings)
    
    # Search for nearest neighbors (k=2 to find oneself and the closest neighbor)
    print("Searching for duplicates...")
    distances, indices = search_index.search(embeddings, k=2)
    
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