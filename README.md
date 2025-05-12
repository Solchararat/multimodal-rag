
# multimodal-rag Documentation

## Overview

`multimodal-rag` is a research and experimentation project that aims to develop a multimodal Retrieval-Augmented Generation (RAG) pipeline capable of enhancing plant identification tasks by grounding Gemini’s AI responses with image-based factual references. The core objective is to bridge image recognition with factual knowledge generation in a robust, verifiable way — combining vision and natural language models.

This solution is designed to support applications like BotaniCatch, an educational and assistive mobile app for plant identification.

## Goals

- Identify plant species from user-submitted images.
- Use retrieved facts from image-text datasets to ground model responses.
- Build an end-to-end multimodal RAG system using Python and open datasets.
- Train and evaluate on real-world biodiversity data from iNaturalist.

## Architecture

User Image + Query
       ↓
   Image Encoder
       ↓
 Text/Vector Embedding ↘
                        → ChromaDB (Vector Store)
       ↑               ↗
  RAG Retriever (FAISS / Chroma) 
       ↓
     Gemini API (RAG Output: Grounded Answer)

## Dataset

### iNaturalist Dataset

- Source: https://www.inaturalist.org
- Type: Public image-text dataset containing observations of flora and fauna with metadata.
- Used Fields:
  - Image files
  - Scientific names
  - Descriptions/annotations
  - Tags and common names
- Usage: Image classification and text embedding generation for factual grounding.

## Technologies Used

| Category        | Tools/Frameworks                          |
|----------------|--------------------------------------------|
| Language        | Python                                     |
| Data Handling   | Pandas, NumPy, OpenCV, PIL                 |
| ML/DL Framework | PyTorch / TensorFlow                      |
| Embeddings      | CLIP / BLIP (Image + Text)                |
| Vector DB       | ChromaDB or FAISS                         |
| LLM             | Gemini API                                |
| Environment     | Google Colab                              |
| Deployment      | Google Cloud Functions (for API calls)    |

## Training & Running

### Setup

pip install -r requirements.txt

### Example Run

```python
from rag_pipeline import run_rag

query = "What is this plant with oval leaves and yellow flowers?"
image_path = "samples/plant_01.jpg"

response = run_rag(image_path, query)
print(response)
```

## Evaluation Metrics

- Top-k Retrieval Accuracy: Measures if the ground-truth plant appears in the top results.
- Answer Grounding Score: Subjectively rates how factual the Gemini-generated answers are.
- BLEU/ROUGE: For text overlap with reference descriptions.

## Use Cases

- Educational plant ID assistant.
- Citizen science tools.
- Smart botany guides in mobile applications.
- Herbarium digitization and classification systems.

## Future Improvements

- Add multilingual support for plant names.
- Integrate feedback loop for retraining embeddings.
- Explore lightweight local models for offline RAG inference.

## Contributors

- Kirsten Dwayne Dizon – AI/ML Speacialist 
- Arron Kian Parejas – Data Scientist & AI Developer  
