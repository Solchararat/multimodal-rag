import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os
from pathlib import Path
from natsort import natsorted
import pandas as pd
import concurrent.futures
from itertools import zip_longest
from tqdm import tqdm
import time

print("Loading ChromaDB client...")
client = chromadb.PersistentClient(path=str(Path(__file__).parent.parent / "db"))

print("Loading dataset...")
df = pd.read_csv("dataset/data.csv")
description = df["description"]
scientific_name = df["scientific_name"]
place = df["place_state_name"]
url = df["url"]
image_url = df["image_url"]

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


def add_image_to_collection(idx, image_path, metadata):
    print(f"\nProcessing image {idx}: {image_path}")
    collection.add(
        ids=[str(idx)],
        uris=[image_path],
        metadatas=[metadata]
    )

print("\nStarting image collection process...")
start_time = time.time()

batches = zip_longest(
    range(len(image_paths)),
    image_paths,
    description,
    scientific_name,
    place,
    url,
    image_url,
    fillvalue=None
)

print("\nSubmitting tasks to executor...")
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for idx, image_path, desc, sci_name, place_name, img_url, url_val in tqdm(batches, total=len(image_paths), desc="Submitting tasks"):
        metadata = {
            "description": desc,
            "scientific_name": sci_name,
            "place": place_name,
            "url": url_val,
            "image_url": img_url
        }
        future = executor.submit(add_image_to_collection, idx, image_path, metadata)
        futures.append(future)
elapsed = time.time() - start_time
print(f"\nCollection created successfully in {elapsed:.2f} seconds")