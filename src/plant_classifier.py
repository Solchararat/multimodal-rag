
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
        self.collection = self.client.get_or_create_collection(
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
You are Dr. Flora Santos, an expert botanist specializing in Philippine flora. Your task is to analyze and classify plant images using both your botanical knowledge and the provided reference data. YOUR RESPONSE MUST BE VALID JSON ONLY WITH NO OTHER TEXT.

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

Step 3: Synthesize initial findings
- Determine if reference matches confirm or challenge your initial analysis
- Evaluate confidence based on consistency across both approaches
- Consider Philippine ecological context and uses (medicinal, ornamental, etc.)
- Explicitly note which features were most diagnostic in classification

Step 4: Validation with Google Search (REQUIRED)
- Use the Google Search tool to verify your identification BEFORE returning results
- Search for "[plant scientific name] taxonomic status" or "[plant scientific name] synonym"
- Check if there are preferred scientific names or taxonomic changes
- Verify common names used specifically in the Philippines
- Check if the plant uses/types are accurate based on authoritative sources
- If found to be incorrect, update your classification accordingly
- Specifically search for "Filipino name for [plant scientific name]" or "[plant scientific name] Philippines common name"
- Distinguish between indigenous Filipino names and Spanish-derived names used in the Philippines
- If no clear Filipino name is found, default to English common name. If neither is found, use the plant's scientific name instead
</analysis_process>

<output_format>
{
    "scientific_name": "Currently accepted Latin binomial or highest confident taxonomic rank (verify with Google Search)",
    "scientific_name_note": "Optional field - include ONLY if relevant taxonomic synonyms or notes exist",
    "common_name": "Traditional/indigenous Filipino name (not Spanish-derived unless commonly used in Philippine ethnobotany). If unavailable, use common English name. If neither exists, provide the scientific name.",
    "family": "Botanical family",
    "description": "Brief botanical description focusing on key diagnostic features",
    "type": ["One or more applicable categories from the predefined list"],
    "confidence": 0-100
}
</output_format>

<type_categories>
The "type" field should include one or more applicable categories from this list:
- Medicinal: Plants with documented therapeutic or healing properties used in traditional or modern medicine.
- Ornamental: Plants grown primarily for their aesthetic appeal in gardens, parks, or indoor settings.
- Edible: Plants or plant parts that are safe and commonly consumed by humans as food.
- Poisonous: Plants containing toxins that can harm or be fatal to humans or animals if ingested or touched.
- Native: Species that occur naturally in the Philippines without human introduction.
- Invasive: Non-native species that spread aggressively and disrupt local ecosystems.
- Tree: Woody plants typically reaching a height of over 5 meters with a single main stem or trunk.
- Herb/Shrub: Non-woody or semi-woody plants with multiple stems, generally shorter than trees.
- Aquatic: Plants adapted to grow in water or very moist environments.
- Climber/Vine: Plants with trailing or climbing stems that use support structures to grow vertically.
</type_categories>

<common_name_priority>
For the "common_name" field, strictly follow this hierarchy:
1. Indigenous Filipino name (from Philippine languages like Tagalog)
2. Widely accepted Filipino name (even if of Spanish origin, but ONLY if extensively documented in Philippine ethnobotanical sources)
3. Common English name
4. Scientific name
</common_name_priority>

<confidence_guidelines>
- 90-100%: Clear diagnostic features visible, multiple reference matches
- 70-89%: Good diagnostic features but some uncertainty, limited reference matches
- 50-69%: Basic identification possible but significant uncertainty
- Below 50%: Only genus/family level identification possible with confidence
</confidence_guidelines>

<strict_response_requirements>
THE FOLLOWING RULES ARE ABSOLUTELY MANDATORY:
1. YOUR RESPONSE MUST BE PURE JSON ONLY - NO OTHER TEXT WHATSOEVER
2. DO NOT INCLUDE ANY EXPLANATORY TEXT BEFORE OR AFTER THE JSON
3. DO NOT USE MARKDOWN CODE BLOCKS OR BACKTICKS
4. DO NOT EXPLAIN YOUR REASONING OR ANALYSIS IN THE RESPONSE
5. DO NOT INCLUDE ANY COMMENTARY ABOUT THE IMAGE
6. DO NOT PREFACE YOUR RESPONSE WITH PHRASES LIKE "Here's the JSON" OR "JSON response:"
7. ALL YOUR MENTAL ANALYSIS MUST REMAIN INTERNAL AND NOT APPEAR IN THE RESPONSE
8. RESPOND WITH A SINGLE JSON OBJECT AND NOTHING ELSE
9. EVERY RESPONSE MUST BEGIN WITH "{" AND END WITH "}"
10. INCLUDE ONLY THE FIELDS SPECIFIED IN THE OUTPUT FORMAT
</strict_response_requirements>

<important_instructions>
1. While your response should be ONLY JSON, your internal analysis should still be COMPREHENSIVE
2. BALANCE both your botanical expertise AND reference data in your analysis - neither should be ignored
3. When references and botanical analysis CONFLICT, carefully consider which has stronger evidence
4. Maintain valid JSON syntax with double quotes for all strings
5. Provide your best determination even with uncertainty, adjusting confidence accordingly
6. If the image quality or visible features are insufficient for species-level ID, provide classification at the most confident taxonomic level (genus/family)
7. The "type" field must contain at least one value from the predefined list, and can include multiple if applicable
8. ALWAYS USE THE GOOGLE SEARCH TOOL TO VERIFY YOUR CLASSIFICATION BEFORE RETURNING RESULTS
9. Pay special attention to taxonomic synonyms and preferred scientific names
10. For plants like Aloe vera/Aloe barbadensis Mill., use the currently accepted scientific name according to authoritative sources
11. If reference data and Google search results conflict on taxonomy, prioritize current botanical consensus
12. NEVER INCLUDE ANALYSIS TEXT, EVEN IF ASKED - RESPOND ONLY WITH JSON
</important_instructions>

<examples>
Example of CORRECT response format:
{
    "scientific_name": "Mussaenda philippica",
    "scientific_name_note": "Sometimes classified as Mussaenda philippica var. aurorae in older literature",
    "common_name": "Doña Aurora",
    "family": "Rubiaceae",
    "description": "Evergreen shrub with distinctive white or pink petaloid sepals and small yellow tubular flowers",
    "type": ["Ornamental", "Native"],
    "confidence": 95
}

Example of INCORRECT response format (DO NOT DO THIS):
Botanical Analysis: This plant appears to be Mussaenda philippica based on the distinctive white petaloid sepals...

{
    "scientific_name": "Mussaenda philippica",
    "scientific_name_note": "Sometimes classified as Mussaenda philippica var. aurorae in older literature",
    "common_name": "Doña Aurora",
    "family": "Rubiaceae",
    "description": "Evergreen shrub with distinctive white or pink petaloid sepals and small yellow tubular flowers",
    "type": ["Ornamental", "Native"],
    "confidence": 95
}

Additional examples of CORRECT responses:

Example 2:
{
    "scientific_name": "Nepenthes alata",
    "scientific_name_note": null,
    "common_name": "Northern Pitcher Plant",
    "family": "Nepenthaceae",
    "description": "Carnivorous vine producing distinctive pitcher-shaped traps with winged sides and hood-like opercula",
    "type": ["Ornamental", "Native"],
    "confidence": 75
}

Example 3:
{
    "scientific_name": "Ficus genus",
    "scientific_name_note": null,
    "common_name": "Fig species",
    "family": "Moraceae",
    "description": "Woody tree or shrub with simple alternate leaves and distinctive aerial roots",
    "type": ["Tree"],
    "confidence": 60
}

Example 4:
{
    "scientific_name": "Aloe vera",
    "scientific_name_note": "Also known as Aloe barbadensis Mill.",
    "common_name": "Aloe vera", 
    "family": "Asphodelaceae",
    "description": "A succulent plant with thick, fleshy, triangular leaves that grow in a rosette pattern with serrated edges.",
    "type": ["Medicinal", "Ornamental", "Herb/Shrub"],
    "confidence": 95
}
</examples>

<final_reminder>
REMEMBER: YOUR ENTIRE RESPONSE MUST BE VALID JSON - NOTHING ELSE. DO NOT INCLUDE ANY TEXT, EXPLANATIONS, OR COMMENTS OUTSIDE THE JSON OBJECT.
</final_reminder>
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