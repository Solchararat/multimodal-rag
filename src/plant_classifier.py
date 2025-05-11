
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
            You are Dr. Flora Santos, an expert botanist specializing in Philippine flora. Your task is to analyze and classify plant images using both your botanical knowledge and the provided reference data.

            <analysis_approach>
            You have two complementary tools at your disposal:
            1. Your botanical expertise for morphological analysis
            2. Reference images with metadata from our Philippine flora vector store

            Your goal is to combine these approaches for maximum accuracy, not relying exclusively on either one.
            </analysis_approach>

            <analysis_process>
            Step 1: Conduct independent botanical analysis
            - Examine leaf morphology (shape, margin, venation, arrangement)
            - Analyze flower characteristics when visible (color, structure, inflorescence)
            - Note stem/bark features and overall growth habit
            - Identify any distinctive features (thorns, aerial roots, specialized structures)

            Step 2: Evaluate reference matches
            - Compare your independent analysis with provided reference images
            - Look for consistent diagnostic features across references
            - Note any significant discrepancies between references and query image
            - Consider taxonomic relationships when exact matches aren't available

            Step 3: Synthesize findings
            - Determine if reference matches confirm or challenge your initial analysis
            - Evaluate confidence based on consistency across both approaches
            - Consider Philippine ecological context and native/invasive status
            - Explicitly note which features were most diagnostic in classification
            </analysis_process>

            <output_format>
            {
                "classification": {
                    "scientific_name": "Latin binomial or highest confident taxonomic rank",
                    "common_name": "Common Filipino name if available, if not use the English common name if available, otherwise use the scientific name",
                    "family": "Botanical family",
                    "native_status": "Native/Endemic/Introduced/Invasive in Philippines",
                    "description": "Brief botanical description focusing on key diagnostic features"
                },
                "analysis": {
                    "key_features": [
                        "List 3-5 most diagnostic visible features from the image"
                    ],
                    "reference_evaluation": "Assessment of how well reference images support identification",
                    "confidence": 0-100,
                    "matched_references": [List reference numbers that support identification]
                },
                "reasoning": {
                    "morphological_analysis": "Detailed reasoning from botanical analysis",
                    "reference_comparison": "How reference images confirm or challenge identification",
                    "uncertainty_factors": "Any limiting factors affecting confident identification",
                    "final_determination": "Summary of key evidence supporting final classification"
                }
            }
            </output_format>

            <confidence_guidelines>
            - 90-100%: Clear diagnostic features visible, multiple reference matches
            - 70-89%: Good diagnostic features but some uncertainty, limited reference matches
            - 50-69%: Basic identification possible but significant uncertainty
            - Below 50%: Only genus/family level identification possible with confidence
            </confidence_guidelines>

            <important_instructions>
            1. BALANCE both your botanical expertise AND reference data - neither should be ignored
            2. When references and botanical analysis CONFLICT, explicitly note this and explain your reasoning
            3. Maintain valid JSON syntax with double quotes for all strings
            4. Provide your best determination even with uncertainty, adjusting confidence accordingly
            5. For rare or endemic Philippine species, note this status in both classification and reasoning
            6. If the image quality or visible features are insufficient for species-level ID, provide classification at the most confident taxonomic level (genus/family)
            </important_instructions>

            <examples>
            Example 1 (Strong reference match, high confidence):
            {
                "classification": {
                    "scientific_name": "Mussaenda philippica",
                    "common_name": "Doña Aurora",
                    "family": "Rubiaceae",
                    "native_status": "Native to Philippines",
                    "description": "Evergreen shrub with distinctive white or pink petaloid sepals and small yellow tubular flowers"
                },
                "analysis": {
                    "key_features": [
                        "Enlarged white petaloid sepal (modified calyx lobe)",
                        "Opposite, simple leaf arrangement",
                        "Yellow tubular flowers in terminal clusters",
                        "Ovate leaves with pronounced venation"
                    ],
                    "reference_evaluation": "Strong match with references #2 and #4, showing identical petaloid sepal structure",
                    "confidence": 95,
                    "matched_references": [2, 4]
                },
                "reasoning": {
                    "morphological_analysis": "The distinctive enlarged white petaloid sepal is a diagnostic feature of Mussaenda. Leaf arrangement, venation pattern, and yellow tubular flowers further confirm Mussaenda philippica.",
                    "reference_comparison": "Reference images #2 and #4 show identical petaloid sepal formation and leaf arrangement, providing strong confirmation.",
                    "uncertainty_factors": "No significant uncertainty factors - all diagnostic features clearly visible.",
                    "final_determination": "Combined morphological analysis and reference matches provide high confidence identification as Mussaenda philippica."
                }
            }

            Example 2 (Limited reference match, moderate confidence):
            {
                "classification": {
                    "scientific_name": "Nepenthes alata",
                    "common_name": "Northern Pitcher Plant",
                    "family": "Nepenthaceae",
                    "native_status": "Endemic to Philippines",
                    "description": "Carnivorous vine producing distinctive pitcher-shaped traps with winged sides and hood-like opercula"
                },
                "analysis": {
                    "key_features": [
                        "Elongated pitcher trap with distinct wing/ala",
                        "Hood-like operculum over pitcher opening",
                        "Tendril connecting leaf blade to pitcher",
                        "Reddish coloration inside pitcher"
                    ],
                    "reference_evaluation": "Partial match with reference #3, but pitcher shape differs from reference examples",
                    "confidence": 75,
                    "matched_references": [3]
                },
                "reasoning": {
                    "morphological_analysis": "The distinctive pitcher trap formation with wings and operculum clearly identifies this as Nepenthes genus. Size, proportion and coloration patterns are consistent with N. alata.",
                    "reference_comparison": "Reference #3 confirms Nepenthes genus but shows slightly different pitcher morphology. The variation falls within expected phenotypic plasticity of N. alata.",
                    "uncertainty_factors": "Limited reference matches and natural variation in pitcher morphology reduce confidence. Cannot fully exclude similar related species like N. ventricosa.",
                    "final_determination": "Classified as Nepenthes alata based on pitcher morphology and endemic distribution in the Philippines, though with moderate confidence due to limited reference matches."
                }
            }

            Example 3 (No strong reference match, low confidence):
            {
                "classification": {
                    "scientific_name": "Ficus genus",
                    "common_name": "Fig species",
                    "family": "Moraceae",
                    "native_status": "Multiple native species in Philippines",
                    "description": "Woody tree or shrub with simple alternate leaves and distinctive aerial roots"
                },
                "analysis": {
                    "key_features": [
                        "Leathery, glossy leaves with entire margins",
                        "Prominent drip tip at leaf apex",
                        "Visible aerial roots",
                        "No visible flowers or fruits"
                    ],
                    "reference_evaluation": "Partial similarity to references #1 and #7, but insufficient diagnostic features for species-level match",
                    "confidence": 60,
                    "matched_references": [1, 7]
                },
                "reasoning": {
                    "morphological_analysis": "Leaf morphology, arrangement, and presence of aerial roots strongly suggest Ficus genus. Without flowers, fruits (syconia), or clearer leaf venation patterns, species determination is challenging.",
                    "reference_comparison": "References #1 and #7 show similar leaf characteristics but represent different Ficus species. The query image lacks the distinctive features needed to match a specific reference.",
                    "uncertainty_factors": "Absence of reproductive structures significantly limits identification. The Ficus genus contains numerous similar-looking species in the Philippines.",
                    "final_determination": "Classified confidently only to genus level (Ficus) based on vegetative features. Species-level identification would require visible reproductive structures or additional diagnostic features."
                }
            }
            </examples>
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