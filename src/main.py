import chromadb
from pathlib import Path
from PIL import Image
import numpy as np
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from itertools import count
from dotenv import load_dotenv
from google.genai import Client
from google.genai.types import  GoogleSearch, Tool, HarmCategory, HarmBlockThreshold, Part, Content, GenerateContentConfig, CreateCachedContentConfig
from os import getenv
import urllib.parse as urlparse
import requests
from io import BytesIO
from typing import Union
import json


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

        
        self.prompt = """
        You are a botanical expert specializing in Philippine flora. Your task is to classify the plant in the query image.
        
        I'll provide you with:
        1. A query image to classify
        2. Several similar reference images with their metadata from our database
        
        Analyze the query image and compare it with the reference examples. Consider:
        - Leaf shape, arrangement, and structure
        - Flower characteristics if visible
        - Overall plant morphology
        - Any distinctive features
        
        Based on this analysis, determine which plant species the query image shows. You can:
        - Confirm it matches one of the reference examples
        - Suggest it's a different species if the characteristics don't match
        - Provide confidence level in your classification
        
        Return a JSON object with the following fields:
        {
            "classification": {
                "scientific_name": "Latin name", 
                "common_name": "Common name", 
                "family": "Plant family"
            },
            "confidence": 0-100,
            "matched_reference": index of the matched reference or null,
            "reasoning": "Brief explanation of key identifying features"
        }
        """
        
        self.cache = self.gemini_client.caches.create(
            model=self.model,
            config=CreateCachedContentConfig(
                display_name="Plant Classifier",
                tools=self.tools,
                system_instruction=self.prompt,
            )
        )
        
        
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

    def classify_with_gemini(self, query_image: str | Image.Image | np.ndarray, similar_images: list[dict[str, Union[Image.Image, dict[str, str]]]]):
        if not isinstance(query_image, bytes):
            _, query_img_bytes, _ = load_image(query_image)
        else:
            query_img_bytes = query_image
        
        example_img_bytes = []
        example_metadata = []
        
        similar_images_iter = iter(similar_images)
        
        for item in similar_images_iter:
            _, img_bytes, _ = load_image(item["image"])
            example_img_bytes.append(img_bytes)
            example_metadata.append(item["metadata"])

        contents = [
            Part.text("\nQUERY IMAGE TO CLASSIFY:"),
            Part.from_bytes(data=query_img_bytes, mime_type="image/jpg")
        ]
        
        for idx, img_bytes, metadata in zip(count(), example_img_bytes, example_metadata):
            contents.extend(
                [
                    Part.text(f"\nREFERENCE IMAGE {idx + 1}:"),
                    Part.from_bytes(data=img_bytes, mime_type="image/jpg"),
                    Part.text(f"\nMETADATA: {json.dumps(metadata, indent=2)}")
                ]
            )
        
        response = self.gemini_client.generate_content(
            model=self.model,
            contents=Content(parts=contents),
            config=GenerateContentConfig(
                cached_content=self.cache,
                safety_settings=self.safety_settings,
            )
        )
        
        try:
            response_text = response.text
            json_start = response_text.find("{")
            json_end = response_text.rfind("}")
            
            if json_start >= 0 and json_end > json_start:
                json_response = response_text[json_start:json_end]
                return json.loads(json_response)
            else:
                return {"raw_response": response_text}
        except Exception as e:
            print(f"Error processing response: {e}")
            return {"error": str(e), "raw_response": response.text}

    def classify_plant(self, query_image: str | Image.Image | np.ndarray, save_results: bool = True, output_dir : str = "dataset/output-images"):

        if save_results:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        similar_images, query_img_array = self.retrieve_similar_images(query_image)
        result = self.classify_with_gemini(query_img_array, similar_images)
        
        if save_results:
            pil_img, _, _ = load_image(query_image)
            pil_img.save(f"{output_dir}/query_image.jpg")
            
            for idx, item in zip(count(), similar_images):
                img = item["image"]
                metadata = item["metadata"]
                scientific_name = metadata.get("scientific_name", f"unknown_{idx}")
                filename = scientific_name.lower().replace(" ", "-") + ".jpg"
                img.save(f"{output_dir}/{idx}-{filename}")

            with open(f"{output_dir}/result.json", "w") as f:
                json.dump(result, f, indent=2)
        
        return result, similar_images

if __name__ == "__main__":
    plant_classifier = PlantClassifier()
    