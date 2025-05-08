import chromadb
from pathlib import Path
from PIL import Image
import numpy as np
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from itertools import count

print("Loading ChromaDB client...")
client = chromadb.PersistentClient(path=str(Path(__file__).parent.parent / "db"))
data_loader = ImageLoader()

embedding_function = OpenCLIPEmbeddingFunction()
print("Loading collection...")

collection = client.get_collection(
    name="philippine_flora",
    embedding_function=embedding_function,
    data_loader=data_loader,
)

print(collection.count())

QUERY_IMG_PATH = "dataset/sample-images/1000_F_175913717_wh9WZV4aT5QAPnJ.jpg"

result = collection.query(
    query_images=[np.array(Image.open(QUERY_IMG_PATH))],
    include=["data", "metadatas"],
    n_results=3
)

for i, (img, metadata) in zip(count(), zip(result['data'][0], result['metadatas'][0])):
    img = Image.fromarray(img)
    scientific_name = metadata.get('scientific_name', f'unknown_{i}')
    filename = scientific_name.lower().replace(' ', '-') + '.jpg'
    img.save(f"dataset/output-images/{i}-{filename}")