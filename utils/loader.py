import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os
from pathlib import Path
from natsort import natsorted
import pandas as pd
from tqdm import tqdm
import time
import re

print("Loading ChromaDB client...")
client = chromadb.PersistentClient(path=str(Path(__file__).parent.parent / "db"))

print("Loading dataset...")
df = pd.read_csv("dataset/data.csv")

print("Loading CLIP model...")
embedding_function = OpenCLIPEmbeddingFunction()

print("Loading image loader...")
image_loader = ImageLoader()

print("Loading collection...")

collection = client.get_or_create_collection(
    name="philippine_flora", embedding_function=embedding_function, data_loader=image_loader,
)

images_dir = Path("dataset") / "images"
image_paths = natsorted([str(images_dir / image).replace("\\", "/") for image in os.listdir(images_dir)])

def extract_index_from_filename(filename):
    match = re.match(r"(\d+)-", os.path.basename(filename))
    if match:
        return int(match.group(1))
    return None

def add_image_to_collection(image_path, metadata):
    idx = extract_index_from_filename(image_path)
    if idx is None:
        idx = "unknown"
    
    existing_ids = collection.get(ids=[str(idx)])
    if existing_ids.get('ids', []):
        return
    
    collection.add(
        ids=[str(idx)],
        uris=[image_path],
        metadatas=[metadata]
    )

print("\nStarting image collection process...")
start_time = time.time()

for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
    idx = extract_index_from_filename(image_path)
    if idx is not None and idx < len(df):
        row = df.iloc[idx]
        metadata = {
            "description": row.get("description", "N/A"),
            "scientific_name": row.get("scientific_name", "N/A"),
            "place": row.get("place_state_name", "N/A"),
            "url": row.get("url", "N/A"),
            "image_url": row.get("image_url", "N/A")
        }
        try:
            add_image_to_collection(image_path, metadata)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    else:
        print(f"Warning: Could not find metadata for image {image_path}")

elapsed = time.time() - start_time
print(f"\nCollection created successfully in {elapsed:.2f} seconds")