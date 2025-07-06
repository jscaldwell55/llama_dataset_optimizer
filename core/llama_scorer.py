# llama_dataset_optimizer/core/llama_scorer.py

import torch
from tqdm import tqdm
from ..utils.llama_utils import apply_chat_template

@torch.no_grad()
def batch_compute_learning_value(dataset, model, tokenizer, batch_size: int):
    """
    Computes a "learning value" score for each sample in the dataset.
    The score is the perplexity of the assistant's response, given the context.
    Higher perplexity means the model was more "surprised" by the response,
    indicating a higher potential learning value.
    """
    print("Scoring samples for learning value (perplexity)...")
    model.eval()
    
    scores = []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Scoring Batches"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        
        batch_scores = []
        for sample in batch:
            # We calculate perplexity on the assistant's turn(s) only
            # The prompt is the context up to the last assistant message
            context_messages = sample['messages'][:-1]
            response_message = sample['messages'][-1]
            
            if response_message['role'] != 'assistant':
                # Skip if the last message is not from the assistant
                batch_scores.append(0.0) 
                continue

            # Format the context and the full conversation
            context_text = tokenizer.apply_chat_template(context_messages, tokenize=False, add_generation_prompt=True)
            full_text = context_text + response_message['content'] + tokenizer.eos_token
            
            context_tokens = tokenizer(context_text, return_tensors='pt').to(model.device)
            full_tokens = tokenizer(full_text, return_tensors='pt').to(model.device)
            
            # The tokens for the response start after the context tokens
            response_start_index = context_tokens.input_ids.shape[1]
            response_end_index = full_tokens.input_ids.shape[1]
            
            # If tokenization results in no new tokens for the response, skip
            if response_start_index >= response_end_index:
                batch_scores.append(0.0)
                continue

            # Get model logits for the entire sequence
            outputs = model(full_tokens.input_ids)
            logits = outputs.logits

            # We only care about the logits for the response part of the sequence
            response_logits = logits[:, response_start_index-1:-1, :]
            response_labels = full_tokens.input_ids[:, response_start_index:]
            
            # Calculate Cross-Entropy Loss for the response
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(response_logits.squeeze(0), response_labels.squeeze(0))
            
            # Perplexity is the exponential of the loss
            perplexity = torch.exp(loss)
            
            # Clamp to avoid extreme values from corrupting the scores
            score = torch.clamp(perplexity, max=10000).item()
            batch_scores.append(score)

        scores.extend(batch_scores)

    # Normalize scores to be between 0 and 1
    max_score = max(scores) if scores else 1.0
    normalized_scores = [s / max_score for s in scores]
    
    return normalized_scores