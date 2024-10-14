# inferencemax/generator.py

import numpy as np
from .utils.decorators import timed
from .utils.logger import debug, info, error

def generate_next_token(max_model, input_ids, attention_mask, past_key_values=None):
    """
    Run the model to get logits for the next token, using past_key_values if provided.
    """
    debug("Generating next token logits")
    try:
        if past_key_values is not None:
            # Only pass the last token and update past_key_values
            outputs = max_model.execute(
                input_ids=input_ids[:, -1:], 
                attention_mask=attention_mask[:, -1:], 
                past_key_values=past_key_values
            )
        else:
            outputs = max_model.execute(input_ids=input_ids, attention_mask=attention_mask)
        
        logits = outputs["logits"].astype(np.float32)
        next_token_logits = logits[:, -1, :]
        new_past_key_values = outputs.get("past_key_values", None)
        return next_token_logits, new_past_key_values
    except Exception as e:
        error(f"Failed to generate next token logits: {str(e)}")
        raise
