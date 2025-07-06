"""
Colab-specific wrapper to avoid flash_attn import issues
"""
import os
import sys

# Disable flash attention before any imports
os.environ['DISABLE_FLASH_ATTN'] = '1'

# Monkey patch to prevent flash_attn import
import transformers.models.llama.modeling_llama
original_import = __builtins__.__import__

def patched_import(name, *args, **kwargs):
    if 'flash_attn' in name:
        raise ImportError(f"flash_attn disabled for compatibility")
    return original_import(name, *args, **kwargs)

__builtins__.__import__ = patched_import

# Now safe to import
from llama_dataset_optimizer import main
from utils.data_formats import DataFormat