# inferencemax/data/onnx.py

import glob
import os
import torch
import onnx
from pathlib import Path
from max import engine
from inferencemax.utils.decorators import timed
from inferencemax.utils.logger import info, warn, error, debug

@timed
def export_onnx_model(model, tokenizer, output_path):
    """Export the model to ONNX format."""
    info(f"Attempting to export model to {output_path}")
    output_path = Path(output_path)
    if output_path.exists():
        warn(f"ONNX model already exists at {output_path}")
        return output_path

    class LogitsModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.logits

    logits_model = LogitsModel(model)
    dummy_input = tokenizer("Example input", return_tensors="pt")
    
    try:
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare inputs
        input_ids = dummy_input["input_ids"]
        attention_mask = dummy_input["attention_mask"]

        # Export the model to ONNX
        torch.onnx.export(
            logits_model,
            args=(input_ids, attention_mask),
            f=str(output_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length", 2: "vocab_size"},
            },
            opset_version=16,
            do_constant_folding=True,
            verbose=False
        )
        info(f"Model successfully exported to ONNX format at {output_path}")
    except Exception as e:
        error(f"Failed to export model: {str(e)}")
        raise
    
    return output_path


def load_onnx_model(model_path):
    """Load an ONNX model."""
    info(f"Loading ONNX model from {model_path}")
    try:
        session = engine.InferenceSession()
        model = session.load(model_path)
        debug("ONNX model loaded successfully")

        # Debug loading
        debug("Model input metadata:")
        for tensor in model.input_metadata:
            debug(f' name:{tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')

        debug("Model output metadata:")
        for tensor in model.output_metadata:
            debug(f'  name: {tensor.name}, shape: {tensor.shape}, dtype: {tensor.dtype}')

        # Call cleanup after successful loading
        cleanup_onnx_artifacts(Path(model_path).stem)

        return model
    except Exception as e:
        error(f"Failed to load ONNX model: {str(e)}")
        raise

@timed
def inspect_onnx_model(onnx_model_path):
    """
    Inspect the ONNX model and print information about its inputs and outputs.
    
    Args:
        onnx_model_path (str or Path): Path to the ONNX model file.
    """
    info(f"Inspecting ONNX model at {onnx_model_path}")
    try:
        onnx_model = onnx.load(onnx_model_path)
        debug("=== Inputs ===")
        for input_tensor in onnx_model.graph.input:
            elem_type = input_tensor.type.tensor_type.elem_type
            elem_type_name = onnx.TensorProto.DataType.Name(elem_type)
            debug(f"{input_tensor.name}: {elem_type_name}")
        debug("=== Outputs ===")
        for output_tensor in onnx_model.graph.output:
            elem_type = output_tensor.type.tensor_type.elem_type
            elem_type_name = onnx.TensorProto.DataType.Name(elem_type)
            debug(f"{output_tensor.name}: {elem_type_name}")
        info("ONNX model inspection complete")
    except Exception as e:
        error(f"Failed to inspect ONNX model: {str(e)}")
        raise

def debug_onnx_model(model_path):
    """
    Debug function to load and inspect an ONNX model.
    
    Args:
        model_path (str or Path): Path to the ONNX model file.
    """
    info("Starting ONNX model debugging process")
    try:
        # First, inspect the model
        inspect_onnx_model(model_path)
        
        # Then, try to load it
        model = load_onnx_model(model_path)
        
        info("ONNX model debugging complete")
        return model
    except Exception as e:
        error(f"An error occurred during ONNX model debugging: {str(e)}")
        raise



def cleanup_onnx_artifacts(model_name):
    """
    Remove temporary ONNX files generated during export.
    
    Args:
        model_name (str): The base name of the model file (without extension)
    """
    debug(f"Starting cleanup of ONNX artifacts for model: {model_name}")
    models_dir = Path("models")
    patterns = [
        "onnx__*",  # Catches all onnx__ prefixed files
        f"{model_name}*.onnx",  # Catches any additional .onnx files
        f"{model_name}.weights",  # Catches the weights file
        "*.onnx_data",  # Catches any onnx_data files
        "onnx_*",  # Catches onnx_ prefixed files
        "*.weights",  # Catches all weight files
        "model.*.weight"  # Catches model weight files
    ]
    
    removed_files = []
    for pattern in patterns:
        full_pattern = models_dir / pattern
        debug(f"Searching for files matching pattern: {full_pattern}")
        for file in glob.glob(str(full_pattern)):
            file_path = Path(file)
            if file_path.name != f"{model_name}.onnx":  # Don't remove the main ONNX file
                try:
                    file_path.unlink()
                    removed_files.append(file_path.name)
                    debug(f"Removed file: {file_path}")
                except Exception as e:
                    warn(f"Failed to remove file {file_path}: {str(e)}")
            else:
                debug(f"Skipping main ONNX file: {file_path}")
    
    if removed_files:
        debug(f"Cleaned up temporary ONNX files: {', '.join(removed_files)}")
    else:
        debug("No temporary ONNX files found to clean up")

    # List remaining files
    remaining_files = list(models_dir.glob("*"))
    debug(f"Remaining files in {models_dir}:")
    for file in remaining_files:
        debug(f"  {file.name}")