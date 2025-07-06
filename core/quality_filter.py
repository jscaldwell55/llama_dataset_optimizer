# llama_dataset_optimizer/core/quality_filter.py

import re
from tqdm import tqdm

def batch_quality_check(dataset, config, tokenizer, batch_size=1000):
    """
    Applies a series of quality checks to each sample in the dataset using batching.
    Returns a list of indices that pass the quality checks.
    """
    print("Applying quality filters...")
    
    passed_indices = []
    cfg = config['quality_filters']
    
    # Process in batches for better performance
    for i in tqdm(range(0, len(dataset), batch_size), desc="Quality Filtering Batches"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        
        # Pre-tokenize the entire batch for efficiency
        batch_messages = [sample['messages'] for sample in batch]
        batch_full_texts = [tokenizer.apply_chat_template(messages, tokenize=False) for messages in batch_messages]
        batch_tokens = tokenizer(batch_full_texts, add_special_tokens=False)
        
        for j, (sample, full_text, tokens) in enumerate(zip(batch, batch_full_texts, batch_tokens['input_ids'])):
            if passes_quality_checks_optimized(sample, cfg, tokenizer, full_text, tokens):
                passed_indices.append(i + j)
            
    print(f"Quality filtering complete. {len(passed_indices)} out of {len(dataset)} samples passed.")
    return passed_indices

def get_total_tokens(messages, tokenizer):
    """Calculate total tokens in a conversation."""
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer.encode(full_text))

def passes_quality_checks_optimized(sample, cfg, tokenizer, full_text, tokens):
    """Optimized version that uses pre-computed tokens and text."""
    messages = sample['messages']
    
    # 1. Turn count
    num_turns = len(messages) // 2 # Assuming alternating user/assistant
    if not (cfg['min_turns'] <= num_turns <= cfg['max_turns']):
        return False

    # 2. Role checks
    roles = [msg['role'] for msg in messages]
    if roles[0] != 'user' or not all(role in cfg['must_have_roles'] for role in roles):
        return False
        
    # 3. Token counts - using pre-computed tokens
    total_tokens = len(tokens)
    if not (cfg['min_total_tokens'] <= total_tokens <= cfg['max_total_tokens']):
        return False
    
    # Count per-turn tokens
    user_tokens = 0
    assistant_tokens = 0
    
    for msg in messages:
        msg_tokens = len(tokenizer.encode(msg['content'], add_special_tokens=False))
        
        if not (cfg['min_tokens_per_turn'] <= msg_tokens <= cfg['max_tokens_per_turn']):
            return False
            
        if msg['role'] == 'user':
            user_tokens += msg_tokens
        elif msg['role'] == 'assistant':
            assistant_tokens += msg_tokens

    # 4. Refusal patterns (only in assistant responses)
    if cfg['check_for_refusal']:
        for msg in messages:
            if msg['role'] == 'assistant':
                for pattern in cfg['refusal_patterns']:
                    if re.search(pattern, msg['content'], re.IGNORECASE):
                        return False

    # 5. Conversation balance
    if user_tokens > 0 and assistant_tokens > 0:
        balance_ratio = abs(user_tokens - assistant_tokens) / (user_tokens + assistant_tokens)
        if balance_ratio > cfg['balanced_conversation_ratio']:
            return False

    return True


def passes_quality_checks(sample, cfg, tokenizer):
    """Checks a single sample against all configured quality filters."""
    messages = sample['messages']
    
    # 1. Turn count
    num_turns = len(messages) // 2 # Assuming alternating user/assistant
    if not (cfg['min_turns'] <= num_turns <= cfg['max_turns']):
        return False

    # 2. Role checks
    roles = [msg['role'] for msg in messages]
    if roles[0] != 'user' or not all(role in cfg['must_have_roles'] for role in roles):
        return False
        
    # 3. Token counts
    total_tokens = 0
    user_tokens = 0
    assistant_tokens = 0
    
    for msg in messages:
        num_tokens = len(tokenizer.encode(msg['content']))
        total_tokens += num_tokens
        
        if not (cfg['min_tokens_per_turn'] <= num_tokens <= cfg['max_tokens_per_turn']):
            return False
            
        if msg['role'] == 'user':
            user_tokens += num_tokens
        elif msg['role'] == 'assistant':
            assistant_tokens += num_tokens

    if not (cfg['min_total_tokens'] <= total_tokens <= cfg['max_total_tokens']):
        return False
        
    # 4. Refusal patterns (only in assistant responses)
    if cfg['check_for_refusal']:
        for msg in messages:
            if msg['role'] == 'assistant':
                for pattern in cfg['refusal_patterns']:
                    if re.search(pattern, msg['content'], re.IGNORECASE):
                        return False

    # 5. Conversation balance
    if user_tokens > 0 and assistant_tokens > 0:
        balance_ratio = abs(user_tokens - assistant_tokens) / (user_tokens + assistant_tokens)
        if balance_ratio > cfg['balanced_conversation_ratio']:
            return False

    return True