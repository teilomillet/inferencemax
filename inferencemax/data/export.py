# inferencemax/data/export.py
from pathlib import Path
from typing import List, Optional
import torch
from inferencemax.utils.decorators import timed
from inferencemax.utils.logger import info, debug, error
from . import onnx

class ConfigurableModel(torch.nn.Module):
    def __init__(self, model, output_attentions=False, output_hidden_states=False):
        super().__init__()
        self.model = model
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states
        )
        return outputs

@timed
def export(
    model, 
    tokenizer, 
    output_path: str, 
    output_attentions: bool = False, 
    output_hidden_states: bool = False,
    debug_mode: bool = False
):
    """
    Export the model to the specified format based on file extension.
    
    Args:
        model: The model to export.
        tokenizer: The tokenizer associated with the model.
        output_path: The path where the exported model will be saved.
        output_attentions: Whether to include attention outputs in the export.
        output_hidden_states: Whether to include hidden state outputs in the export.
        debug_mode: Whether to run in debug mode, which includes model inspection.
    """
    info(f"Starting model export process to {output_path}")
    
    output_path = Path(output_path)
    format = output_path.suffix.lower()[1:]  # Remove the leading dot
    
    debug(f"Detected export format: {format}")
    
    try:
        if format == 'onnx':
            info("Exporting model to ONNX format")
            wrapped_model = ConfigurableModel(model, output_attentions, output_hidden_states)
            result = onnx.export_onnx_model(wrapped_model, tokenizer, output_path)
            
            if debug_mode:
                info("Running in debug mode. Inspecting the exported ONNX model.")
                onnx.debug_onnx_model(output_path)
            
            info(f"Model successfully exported to {output_path}")
            return result
        else:
            error(f"Unsupported export format: {format}")
            raise ValueError(f"Unsupported export format: {format}")
    except Exception as e:
        error(f"An error occurred during model export: {str(e)}")
        raise