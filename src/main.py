import chromadb
from pathlib import Path
from PIL import Image
import numpy as np
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from itertools import count
from dotenv import load_dotenv
from google.genai import Client
from google.genai.types import  GoogleSearch, Tool, HarmCategory, HarmBlockThreshold
from os import getenv
import urllib.parse as urlparse
import requests
from io import BytesIO
from typing import Union
load_dotenv()

MODEL_ID = "gemini-2.5-flash-preview-04-17"

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

def is_url(string: str) -> bool: 
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False    

def load_image(image_input: str | Image.Image | np.ndarray) -> tuple[Image.Image, bytes]:
    img_bytes: bytes | None = None
    
    if isinstance(image_input, str):
        if is_url(image_input):
            response = requests.get(image_input, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img_bytes = response.content
            img_array = np.array(img)
        else: # local file path
            img = Image.open(image_input)
            img_array = np.array(img)
            with open(image_input, "rb") as f:
                img_bytes = f.read()
    elif isinstance(image_input, Image.Image):
        img = image_input
        img_array = np.array(img)    
        buffer = BytesIO()
        img.save(buffer, format="jpg")
        buffer.seek(0)
        img_bytes = buffer.get_value()
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
        img_array = image_input
        buffer = BytesIO()
        img.save(buffer, format="jpg")
        buffer.seek(0)
        img_bytes = buffer.get_value()

    return img, img_bytes, img_array
class PlantClassifier:
    def __init__(self, db_path: str, collection_name: str ="philippine_flora"):
        print("Loading ChromaDB client...")
        self.client = chromadb.PersistentClient(path=str(Path(db_path)))

        print("Loading image loader...")
        self.data_loader = ImageLoader()

        print("Loading CLIP model...")
        self.embedding_function = OpenCLIPEmbeddingFunction()

        print("Loading collection...")
        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            data_loader=self.data_loader,
        )
        
        self.gemini_client = Client(
            api_key=getenv("GEMINI_API_KEY")
        )
        
        self.model = MODEL_ID
        self.tools = [Tool(google_search=GoogleSearch()),]
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_VIOLENCE: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SELF_HARM: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

    def retrieve_similar_images(self, query_image_input: str | Image.Image | np.ndarray, n_results: int = 5) -> tuple[list[dict[str, Union[Image.Image, dict[str, str]]]], np.ndarray]:
        _, _, query_img_array = load_image(query_image_input)

        result = self.collection.query(
            query_images=[query_img_array],
            include=["data", "metadatas"],
            n_results=n_results
        )

        images = []
        metadatas = result["metadatas"][0]
        data = result["data"][0]
        
        for img, metadata in zip(data, metadatas):
            pil_img = Image.fromarray(img)
            images.append(
                {
                    "image": pil_img,
                    "metadata": metadata
                }
            )

        return images, query_img_array

if __name__ == "__main__":
    plant_classifier = PlantClassifier()
    