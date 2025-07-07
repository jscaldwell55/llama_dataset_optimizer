# llama_dataset_optimizer/utils/llama_utils.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(
    model_name: str, 
    use_4bit: bool = False,
    use_flash_attn: bool = False
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads a Llama model and its tokenizer with optional optimizations.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model '{model_name}'...")
    logger.info(f"Configuration: use_4bit={use_4bit}, use_flash_attn={use_flash_attn}")
    
    try:
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise

    # Determine the best device
    if torch.cuda.is_available():
        device_map = "auto"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device_map = {"": "mps"}
        dtype = torch.float16  # MPS works better with float16
    else:
        device_map = {"": "cpu"}
        dtype = torch.float32  # CPU needs float32
    
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
    }
    
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            logger.info("Using 4-bit quantization")
        except ImportError:
            logger.warning("bitsandbytes not available, skipping 4-bit quantization")
            use_4bit = False

    # To enable flash attention, the model must be loaded with `attn_implementation="flash_attention_2"`
    # and the library must be installed (`pip install flash-attn`).
    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("Using Flash Attention 2.")

    try:
        logger.info(f"Loading model with kwargs: {model_kwargs}")
        logger.info("Attempting to load model from pretrained...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            **model_kwargs,
            local_files_only=False,
            low_cpu_mem_usage=True
        )
        logger.info("Model loaded into memory")
    except ImportError as e:
        if "flash_attn" in str(e) and use_flash_attn:
            logger.warning("Flash Attention 2 not available. Falling back to standard attention.")
            model_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                **model_kwargs,
                local_files_only=False,
                low_cpu_mem_usage=True
            )
        else:
            logger.error(f"Import error while loading model: {e}")
            raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    logger.info(f"Model '{model_name}' loaded successfully on device: {model.device}")
    return model, tokenizer

def apply_chat_template(sample, tokenizer):
    """Applies the chat template to a sample's messages."""
    return tokenizer.apply_chat_template(
        sample['messages'], 
        tokenize=False, 
        add_generation_prompt=False
    )