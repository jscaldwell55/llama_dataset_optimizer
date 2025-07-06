# llama_dataset_optimizer/core/llama_scorer.py

import torch
from tqdm import tqdm
from utils.llama_utils import apply_chat_template

@torch.no_grad()
def batch_compute_learning_value(dataset, model, tokenizer, batch_size: int):
    """
    Computes a "learning value" score for each sample in the dataset using true batch processing.
    The score is the perplexity of the assistant's response, given the context.
    Higher perplexity means the model was more "surprised" by the response,
    indicating a higher potential learning value.
    """
    print("Scoring samples for learning value (perplexity)...")
    model.eval()
    
    scores = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Scoring Batches"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        
        # Prepare batch data
        batch_contexts = []
        batch_full_texts = []
        batch_response_lengths = []
        valid_indices = []
        
        for j, sample in enumerate(batch):
            # We calculate perplexity on the assistant's turn(s) only
            context_messages = sample['messages'][:-1]
            response_message = sample['messages'][-1]
            
            if response_message['role'] != 'assistant':
                continue
                
            # Format the context and the full conversation
            context_text = tokenizer.apply_chat_template(context_messages, tokenize=False, add_generation_prompt=True)
            full_text = context_text + response_message['content'] + tokenizer.eos_token
            
            batch_contexts.append(context_text)
            batch_full_texts.append(full_text)
            valid_indices.append(j)
        
        if not batch_contexts:
            # No valid samples in this batch
            scores.extend([0.0] * len(batch))
            continue
            
        # Tokenize in batches
        context_tokens = tokenizer(batch_contexts, return_tensors='pt', padding=True, truncation=True, max_length=2048).to(model.device)
        full_tokens = tokenizer(batch_full_texts, return_tensors='pt', padding=True, truncation=True, max_length=2048).to(model.device)
        
        # Get model logits for the entire batch
        outputs = model(full_tokens.input_ids)
        logits = outputs.logits
        
        # Calculate perplexity for each sample in the batch
        batch_scores = [0.0] * len(batch)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        for j, valid_idx in enumerate(valid_indices):
            ctx_tokens = context_tokens.input_ids[j]
            full_tokens_sample = full_tokens.input_ids[j]
            
            # Find actual response start (accounting for padding)
            ctx_len = (ctx_tokens != tokenizer.pad_token_id).sum().item()
            full_len = (full_tokens_sample != tokenizer.pad_token_id).sum().item()
            response_start = ctx_len
            response_end = full_len
            
            if response_start >= response_end:
                continue
                
            # Extract response logits and labels
            response_logits = logits[j, response_start-1:response_end-1, :]
            response_labels = full_tokens_sample[response_start:response_end]
            
            # Calculate loss for this response
            losses = loss_fct(response_logits, response_labels)
            avg_loss = losses.mean()
            
            # Perplexity is the exponential of the loss
            perplexity = torch.exp(avg_loss)
            score = torch.clamp(perplexity, max=10000).item()
            
            batch_scores[valid_idx] = score
        
        scores.extend(batch_scores)

    # Normalize scores to be between 0 and 1
    max_score = max(scores) if scores else 1.0
    normalized_scores = [s / max_score for s in scores]
    
    return normalized_scores