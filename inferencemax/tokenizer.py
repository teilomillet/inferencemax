# inferencemax/tokenizer/tokenizer.py

import numpy as np
from typing import Union, List
from max.dtype import DType as max_DType
from inferencemax.utils.decorators import timed
from inferencemax.utils.logger import debug, info, error


def tokenize_input(tokenizer, input_text: Union[str, List[str]]):
    """
    Tokenize the input text using the tokenizer.

    Args:
        tokenizer: The tokenizer to use.
        input_text (str or list of str): The input prompt(s).

    Returns:
        input_ids (np.ndarray): Tokenized input IDs.
        attention_mask (np.ndarray): Attention mask for the input.
    """
    debug(f"Tokenizing input: {input_text[:50]}...")
    try:
        inputs = tokenizer(input_text, return_tensors="np", padding=True, truncation=True)
        input_ids = inputs["input_ids"].astype(max_DType.int64.to_numpy())
        attention_mask = inputs["attention_mask"].astype(max_DType.int64.to_numpy())
        info(f"Input tokenized successfully. Shape: {input_ids.shape}")
        return input_ids, attention_mask
    except Exception as e:
        error(f"Failed to tokenize input: {str(e)}")
        raise


def decode_output(tokenizer, input_ids):
    """
    Decode the generated token IDs back into text.

    Args:
        tokenizer: The tokenizer to use.
        input_ids (np.ndarray): Generated input IDs.

    Returns:
        generated_texts (List[str]): Decoded texts.
    """
    debug("Decoding output tokens...")
    try:
        generated_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        debug(f"Output decoded successfully. Number of texts: {len(generated_texts)}")
        return generated_texts
    except Exception as e:
        error(f"Failed to decode output: {str(e)}")
        raise