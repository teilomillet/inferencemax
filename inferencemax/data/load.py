# inferencemax/data/load.py
from pathlib import Path
from inferencemax.utils.decorators import timed
from inferencemax.utils.logger import debug, info, error
from . import onnx, hf


@timed
def load_model(model_path: str, model_type: str = "auto"):
    """
    Load a model from the specified path.
    
    Args:
        model_path: The path to the model file or HF model name.
        model_type: The type of model to load ("auto", "onnx", or "hf").
    
    Returns:
        The loaded model.
    """
    debug(f"Attempting to load model from: {model_path}")
    debug(f"Specified model type: {model_type}")

    if model_type == "auto":
        model_path = Path(model_path)
        if model_path.exists():
            format = model_path.suffix.lower()[1:]
            if format == 'onnx':
                model_type = "onnx"
            else:
                model_type = "hf"
        else:
            model_type = "hf"
        debug(f"Auto-detected model type: {model_type}")

    try:
        if model_type == "onnx":
            model = onnx.load_onnx_model(model_path)
        elif model_type == "hf":
            model = hf.load_hf_model(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        info(f"Successfully loaded {model_type} model from {model_path}")
        return model
    except Exception as e:
        error(f"Failed to load model from {model_path}. Error: {str(e)}")
        raise

@timed
def load_tokenizer(model_path: str):
    """
    Load a tokenizer for the specified model.
    
    Args:
        model_path: The path to the model file or HF model name.
    
    Returns:
        The loaded tokenizer.
    """
    debug(f"Attempting to load tokenizer for model: {model_path}")
    try:
        tokenizer = hf.load_hf_tokenizer(model_path)
        debug(f"Successfully loaded tokenizer for {model_path}")
        return tokenizer
    except Exception as e:
        error(f"Failed to load tokenizer for {model_path}. Error: {str(e)}")
        raise