# llama_dataset_optimizer/core/quality_filter.py

import re
from tqdm import tqdm

def batch_quality_check(dataset, config, tokenizer):
    """
    Applies a series of quality checks to each sample in the dataset.
    Returns a list of boolean values (True for pass, False for fail).
    """
    print("Applying quality filters...")
    
    passed_indices = []
    cfg = config['quality_filters']
    
    for i, sample in enumerate(tqdm(dataset, desc="Quality Filtering")):
        if passes_quality_checks(sample, cfg, tokenizer):
            passed_indices.append(i)
            
    print(f"Quality filtering complete. {len(passed_indices)} out of {len(dataset)} samples passed.")
    return passed_indices

def get_total_tokens(messages, tokenizer):
    """Calculate total tokens in a conversation."""
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer.encode(full_text))

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