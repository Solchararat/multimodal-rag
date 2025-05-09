
import chromadb
from PIL import Image
import numpy as np
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from itertools import count
from dotenv import load_dotenv
from google.genai import Client
from google.genai.types import  GoogleSearch, Tool, HarmCategory, HarmBlockThreshold, Part, Content, GenerateContentConfig, SafetySetting
from os import getenv
import urllib.parse
import requests
from io import BytesIO
from typing import Union
from pathlib import Path
import json

class PlantClassifier:
    def __init__(self, db_path: str, collection_name: str ="philippine_flora"):
        load_dotenv()
        
        MODEL_ID = "gemini-2.5-flash-preview-04-17"
        
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
        
        self.safety_settings = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
            )
        ]
        
        self.prompt = """
            You are a botanical expert specializing in Philippine flora. Your task is to classify the plant in the query image using only the provided reference data.

            I'll provide:
            1. A query image to classify
            2. Several reference images with metadata

            Analysis steps:
            1. Examine leaf morphology (shape, arrangement, venation)
            2. Note flower characteristics (if visible)
            3. Compare overall structure and distinctive features
            4. Match against reference specimens

            Respond STRICTLY in this JSON format:
            {
                "classification": {
                    "scientific_name": "Latin name", 
                    "common_name": "Common name",
                    "family": "Plant family",
                    "description": "Plant characteristics"
                },
                "confidence": 0-100,
                "matched_reference": Number of matched reference images,
                "reasoning": "Key distinguishing features"
            }

            Examples of valid responses:

            Example 1 (Confident match):
            {
                "classification": {
                    "scientific_name": "Mussaenda philippica",
                    "common_name": "Aguho",
                    "family": "Rubiaceae",
                    "description": "Evergreen shrub with oval leaves and showy white bracts surrounding small yellow flowers"
                },
                "confidence": 95,
                "matched_reference": 2,
                "reasoning": "Matched reference #2 through distinct white petaloid bracts and opposite leaf arrangement"
            }

            Example 2 (Different species):
            {
                "classification": {
                    "scientific_name": "Nepenthes attenboroughii",
                    "common_name": "Attenborough's Pitcher Plant",
                    "family": "Nepenthaceae",
                    "description": "Carnivorous vine with elongated pitcher traps and lance-shaped leaves"
                },
                "confidence": 82,
                "matched_reference": null,
                "reasoning": "Pitcher morphology differs from references - wider peristome and more pronounced lid suggests different Nepenthes species"
            }

            Example 3 (Low confidence):
            {
                "classification": {
                    "scientific_name": "Ficus pseudopalma",
                    "common_name": "Philippine fig palm",
                    "family": "Moraceae",
                    "description": "Palm-like tree with terminal cluster of fiddle-shaped leaves and aerial roots"
                },
                "confidence": 65,
                "matched_reference": 4,
                "reasoning": "Partial match to reference #4 in leaf shape, but unclear stem characteristics reduce confidence"
            }

            Important:
            - ALWAYS maintain valid JSON syntax
            - Use double quotes for all strings
            - Never include markdown formatting
            - Omit any explanatory text outside the JSON
            - If uncertain, provide best classification attempt with adjusted confidence
            """
    
    def is_url(self, string: str) -> bool: 
        try:
            result = urllib.parse.urlparse(string)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False    
    
    def load_image(self, image_input: str | Image.Image | np.ndarray) -> tuple[Image.Image, bytes]:
        img_bytes: bytes | None = None
        
        if isinstance(image_input, str):
            if self.is_url(image_input):
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
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            img_bytes = buffer.getvalue()
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
            img_array = image_input
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)
            img_bytes = buffer.getvalue()

        return img, img_bytes, img_array
        
    def retrieve_similar_images(self, query_image_input: str | Image.Image | np.ndarray, n_results: int = 5) -> tuple[list[dict[str, Union[Image.Image, dict[str, str]]]], np.ndarray]:
        _, _, query_img_array = self.load_image(query_image_input)

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
            _, query_img_bytes, _ = self.load_image(query_image)
        else:
            query_img_bytes = query_image
        
        example_img_bytes = []
        example_metadata = []
        
        similar_images_iter = iter(similar_images)
        
        for item in similar_images_iter:
            _, img_bytes, _ = self.load_image(item["image"])
            example_img_bytes.append(img_bytes)
            example_metadata.append(item["metadata"])

        contents = [
            Part.from_text(text=self.prompt),
            Part.from_text(text="\nQUERY IMAGE TO CLASSIFY:"),
            Part.from_bytes(data=query_img_bytes, mime_type="image/jpeg")
        ]
        
        for idx, img_bytes, metadata in zip(count(), example_img_bytes, example_metadata):
            contents.extend(
                [
                    Part.from_text(text=f"\nREFERENCE IMAGE {idx + 1}:"),
                    Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    Part.from_text(text=f"\nMETADATA: {json.dumps(metadata, indent=2)}")
                ]
            )
        
        try:
            response = self.gemini_client.models.generate_content(
                model=self.model,
                contents=Content(parts=contents),
                config=GenerateContentConfig(
                    tools=self.tools,
                    safety_settings=self.safety_settings,
                )
            )
            text = response.text.strip()
            try:
                if text.startswith("```"):
                    text = text.strip("` \n")
                    if text.lower().startswith("json"):
                        text = text[4:].lstrip("\n")
                parsed = json.loads(text)
                return parsed
            except Exception as parse_err:
                print(f"Failed to parse Gemini response as JSON: {parse_err}\nRaw response: {text}")
                return {"error": f"Failed to parse Gemini response as JSON: {parse_err}", "raw_response": text}
        except Exception as e:
            print(f"Error processing response: {e}")
            return {"error": str(e)}

    def classify_plant(self, query_image: str | Image.Image | np.ndarray, save_results: bool = True, output_dir : str = "dataset/output-images"):

        if save_results:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        similar_images, query_img_array = self.retrieve_similar_images(query_image)
        result = self.classify_with_gemini(query_img_array, similar_images)
        
        if save_results:
            pil_img, _, _ = self.load_image(query_image)
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