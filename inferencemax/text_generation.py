# inferencemax/text_generation.py

from pathlib import Path
import glob
from tqdm import tqdm
from typing import Union, List
from .utils.decorators import timed
from .utils.logger import debug, info, error
from .tokenizer import tokenize_input, decode_output
from .initializer import initialize_generation, update_inputs
from .generator import generate_next_token
from .sampler import process_logits, sample_next_token
from .kv_cache import initialize_kv_cache, update_kv_cache

@timed
def generate_text(
    max_model,
    tokenizer,
    input_text: Union[str, List[str]],
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_k: int = 50,
):
    debug(f"Starting text generation for input: {input_text[:50]}...")
    try:
        input_ids, attention_mask = tokenize_input(tokenizer, input_text)
        input_ids, attention_mask, batch_size = initialize_generation(input_ids, attention_mask)

        past_key_values = initialize_kv_cache()

        with tqdm(total=max_new_tokens, desc="Generating tokens") as pbar:
            for _ in range(max_new_tokens):
                next_token_logits, past_key_values = generate_next_token(
                    max_model, input_ids, attention_mask, past_key_values=past_key_values
                )
                top_k_indices, top_k_probs = process_logits(next_token_logits, temperature, top_k)
                next_tokens = sample_next_token(top_k_indices, top_k_probs)
                input_ids, attention_mask = update_inputs(input_ids, attention_mask, next_tokens)
                pbar.update(1)

        generated_texts = decode_output(tokenizer, input_ids)
        info(f"Text generation complete. Generated {len(generated_texts)} texts.")

        return generated_texts
    except Exception as e:
        error(f"Text generation failed: {str(e)}")
        raise
