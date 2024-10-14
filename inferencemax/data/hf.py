# inferencemax/data/hf.py

from inferencemax.utils.decorators import timed
from inferencemax.utils.logger import debug, info, warn, error
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_hf_tokenizer(model_name_or_path: str):
    """
    Load the tokenizer from the specified Hugging Face model name or path.
    Args:
        model_name_or_path (str): The model name or local path.
    Returns:
        tokenizer: The loaded tokenizer.
    """
    debug(f"Attempting to load HF tokenizer from: {model_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        debug(f"Successfully loaded HF tokenizer from: {model_name_or_path}")
        
        if tokenizer.pad_token is None:
            debug("Pad token not set. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            info("Set pad_token to eos_token")
        else:
            debug("Pad token already set")
        
        return tokenizer
    except Exception as e:
        error(f"Failed to load HF tokenizer from {model_name_or_path}. Error: {str(e)}")
        raise

def load_hf_model(model_name_or_path: str, torch_dtype=torch.float16):
    """
    Load the model from the specified Hugging Face model name or path.
    Args:
        model_name_or_path (str): The model name or local path.
        torch_dtype: The data type for the model weights.
    Returns:
        model: The loaded model.
    """
    debug(f"Attempting to load HF model from: {model_name_or_path}")
    debug(f"Using torch dtype: {torch_dtype}")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        )
        debug(f"Successfully loaded HF model from: {model_name_or_path}")
        
        model.eval()
        debug("HF Model set to evaluation mode")
        
        return model
    except Exception as e:
        error(f"Failed to load HF model from {model_name_or_path}. Error: {str(e)}")
        raise