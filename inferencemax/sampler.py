# inferencemax/sampler.py

import numpy as np
from .utils.decorators import timed
from .utils.logger import debug, info, error


def process_logits(next_token_logits, temperature, top_k):
    """
    Apply temperature scaling and top-k sampling to the logits.
    """
    debug(f"Processing logits with temperature {temperature} and top_k {top_k}")
    try:
        next_token_logits = next_token_logits / temperature
        top_k_indices = np.argpartition(next_token_logits, -top_k, axis=-1)[:, -top_k:]
        top_k_logits = np.take_along_axis(next_token_logits, top_k_indices, axis=-1)
        top_k_logits = top_k_logits - np.max(top_k_logits, axis=-1, keepdims=True)
        top_k_probs = np.exp(top_k_logits)
        top_k_probs /= np.sum(top_k_probs, axis=-1, keepdims=True)
        return top_k_indices, top_k_probs
    except Exception as e:
        error(f"Failed to process logits: {str(e)}")
        raise


def sample_next_token(top_k_indices, top_k_probs):
    """
    Sample the next token from the top-k probabilities.
    """
    debug("Sampling next token")
    try:
        batch_size = top_k_indices.shape[0]
        cumulative_probs = np.cumsum(top_k_probs, axis=-1)
        random_values = np.random.rand(batch_size, 1)
        next_token_positions = (cumulative_probs > random_values).argmax(axis=-1)
        next_tokens = np.take_along_axis(
            top_k_indices, next_token_positions[:, None], axis=-1
        ).reshape(-1)
        return next_tokens
    except Exception as e:
        error(f"Failed to sample next token: {str(e)}")
        raise