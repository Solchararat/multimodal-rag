import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from pathlib import Path
from PIL import Image
import numpy as np

print("Loading ChromaDB client...")
client = chromadb.PersistentClient(path=str(Path(__file__).parent.parent / "db"))

print("Loading CLIP model...")
embedding_function = OpenCLIPEmbeddingFunction()

print("Loading image loader...")
image_loader = ImageLoader()

print("Loading collection...")
collection = client.get_or_create_collection(
    name="philippine_flora", embedding_function=embedding_function, data_loader=image_loader,
)

QUERY_IMG_PATH = "dataset/sample-images/Diplaziumcordifolium1.png"

result = collection.query(
    query_images=[np.array(Image.open(QUERY_IMG_PATH))],
    include=["data"],
    n_results=3
)


i = 0

for img in result['data'][0]:
    img = Image.fromarray(img)
    img.save(f"dataset/output-images/{i}.jpg")
    i += 1