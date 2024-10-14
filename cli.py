# inferencemax/cli.py
import os
import fire
import yaml
from pathlib import Path
from inferencemax.data.load import load_model, load_tokenizer
from inferencemax.data.export import export
from inferencemax.text_generation import generate_text

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run(model_path, input_text, config_path=None, **kwargs):
    if config_path:
        config = load_config(config_path)
        config.update(kwargs)  # Override with any provided kwargs
    else:
        config = kwargs

    # Create models directory if it doesn't exist
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    # Define ONNX model path
    onnx_model_path = models_dir / f"{Path(model_path).name.replace('/', '_')}.onnx"

    # Load tokenizer
    tokenizer = load_tokenizer(model_path)

    if not onnx_model_path.exists():
        # Load HuggingFace model
        hf_model = load_model(model_path, model_type="hf")
        
        # Export to ONNX
        export(hf_model, tokenizer, str(onnx_model_path))
    else:
        print(f"ONNX model already exists at {onnx_model_path}, skipping model loading and export.")

    # Load the ONNX model with MAX Engine
    max_model = load_model(str(onnx_model_path), model_type="onnx")

    generated_text = generate_text(
        max_model,
        tokenizer,
        input_text,
        max_new_tokens=config.get('max_new_tokens', 32),
        temperature=config.get('temperature', 0.7),
        top_k=config.get('top_k', 50),
    )

    print(f"Generated text: {generated_text}")

def main():
    fire.Fire(run)

if __name__ == "__main__":
    main()