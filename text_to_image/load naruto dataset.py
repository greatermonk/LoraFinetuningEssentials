from datasets import load_dataset
import os
import json
from PIL import Image
import base64
from io import BytesIO

def serialize_example(example):
    serialized = {}
    for key, value in example.items():
        if isinstance(value, Image.Image):
            # Convert image to base64-encoded string
            buffered = BytesIO()
            value.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            serialized[key] = {
                "image_data": img_str,
                "image_format": "PNG"
            }
        elif isinstance(value, (int, float, str, bool, type(None))):
            serialized[key] = value
        else:
            # For other complex types, convert to string
            serialized[key] = str(value)
    return serialized

def download_and_save_dataset(dataset_name, subset=None, split="train", num_samples=None, output_dir="downloaded_dataset"):
    # Load the dataset
    dataset = load_dataset(dataset_name, subset, split=split)
    
    # Take a subset if specified
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each example as a separate JSON file
    for i, example in enumerate(dataset):
        file_path = os.path.join(output_dir, f"example_{i}.json")
        serialized_example = serialize_example(example)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serialized_example, f, ensure_ascii=False, indent=2)

    print(f"Dataset saved to {output_dir}")

# Example usage
download_and_save_dataset("lambdalabs/naruto-blip-captions", split="train", num_samples=100, output_dir="naruto_dataset")