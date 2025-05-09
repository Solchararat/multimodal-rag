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

def load_image_from_path_or_url(image_path_or_url: str) -> tuple[Image.Image, bytes]:
    if is_url(image_path_or_url):
        response = requests.get(image_path_or_url, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        img_bytes = response.content
    else:
        image = Image.open(image_path_or_url)
        with open(image_path_or_url, "rb") as f:
            img_bytes = f.read()
    
    return image, img_bytes
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

if __name__ == "__main__":
    plant_classifier = PlantClassifier()
    