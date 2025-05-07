import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os
from pathlib import Path
from natsort import natsorted
import pandas as pd
client = chromadb.Client()

df = pd.read_csv("dataset/data.csv")


description = df["description"]
scientific_name = df["scientific_name"]
place = df["place_state_name"]
url = df["url"]
image_url = df["image_url"]


embedding_function = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()

collection = client.get_or_create_collection(
    name="philippine_flora", embedding_function=embedding_function, data_loader=image_loader,
)

images_dir = Path("dataset") / "images"
image_paths = natsorted([str(images_dir / image).replace("\\", "/") for image in os.listdir(images_dir)])

for i in range(len(image_paths)):
    collection.add(ids=[str(i)], uris=[image_paths[i]], metadatas=[{
        "description": description[i],
        "scientific_name": scientific_name[i],
        "place": place[i],
        "url": url[i],
        "image_url": image_url[i]
    }])

print("Collection created successfully.")
