# inferencemax/initializer.py

import numpy as np
from max.dtype import DType as max_DType
from .utils.logger import debug, info, error


def initialize_generation(input_ids, attention_mask):
    """
    Initialize variables required for text generation.
    """
    batch_size = input_ids.shape[0]
    debug(f"Initializing generation with batch size: {batch_size}")
    return input_ids, attention_mask, batch_size


def initialize_sampling_parameters(temperature: float = 0.7, top_k: int = 50):
    """
    Initialize parameters for sampling during generation.
    """
    debug(f"Initializing sampling parameters: temperature={temperature}, top_k={top_k}")
    return {"temperature": temperature, "top_k": top_k}


def update_inputs(input_ids, attention_mask, next_tokens):
    """
    Update the input IDs and attention mask with the next tokens.
    """
    debug("Updating inputs with next tokens")
    try:
        input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=1)
        new_attention = np.ones((input_ids.shape[0], 1), dtype=max_DType.int64.to_numpy())
        attention_mask = np.concatenate([attention_mask, new_attention], axis=1)
        return input_ids, attention_mask
    except Exception as e:
        error(f"Failed to update inputs: {str(e)}")
        raise

# Add more