# llama_dataset_optimizer/utils/llama_utils.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(
    model_name: str, 
    use_4bit: bool = False,
    use_flash_attn: bool = True
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a Llama model and its tokenizer with optional optimizations.
    """
    print(f"Loading model '{model_name}'...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    # To enable flash attention, the model must be loaded with `attn_implementation="flash_attention_2"`
    # and the library must be installed (`pip install flash-attn`).
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using Flash Attention 2.")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except ImportError as e:
        if "flash_attn" in str(e) and use_flash_attn:
            print("Flash Attention 2 not available. Falling back to standard attention.")
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        else:
            raise
    
    print(f"Model '{model_name}' loaded successfully on device: {model.device}")
    return model, tokenizer

def apply_chat_template(sample, tokenizer):
    """Applies the chat template to a sample's messages."""
    return tokenizer.apply_chat_template(
        sample['messages'], 
        tokenize=False, 
        add_generation_prompt=False
    )