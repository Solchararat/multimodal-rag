
# multimodal-rag Documentation

## 📌 Overview

`multimodal-rag` is a research and experimentation project that aims to develop a multimodal Retrieval-Augmented Generation (RAG) pipeline capable of enhancing plant identification tasks by grounding Gemini’s AI responses with image-based factual references. The core objective is to bridge image recognition with factual knowledge generation in a robust, verifiable way — combining vision and natural language models.

This solution is designed to support applications like BotaniCatch, an educational and assistive mobile app for plant identification.

## 🥅 Goals

- Identify plant species from user-submitted images.
- Use retrieved facts from image-text datasets to ground model responses.
- Build an end-to-end multimodal RAG system using Python and open datasets.
- Train and evaluate on real-world biodiversity data from iNaturalist.

## ⚙️ Architecture

User Image + Query
       ↓
  Image Encoder
       ↓
ChromaDB (Vector Store)
       ↓
 Vector Embedding
       ↓
Gemini API (RAG Output: Grounded Answer)

## 📊 Dataset

### 🌱 iNaturalist Dataset

- Source: https://www.inaturalist.org
- Type: Public image-text dataset containing observations of flora and fauna with metadata.
- Used Fields:
  - Image files
  - Scientific names
  - Descriptions/annotations
  - Tags and common names
- Usage: Image classification and text embedding generation for factual grounding.

## 📱 Technologies Used

| Category        | Tools/Frameworks                          |
|-----------------|-------------------------------------------|
| Language        | Python                                    |
| Data Handling   | Pandas, NumPy, OpenCV, PIL                |
| ML/DL Framework | PyTorch                                   |
| Embeddings      | CLIP                                      |
| Vector DB       | ChromaDB                                  |
| LLM             | Gemini API                                |
| Deployment      | Google Cloud Run (for API calls)          |

## 🏃 Running

### ⚒️ Setup

pip install -r requirements.txt

Run the `downloader.py` to download the plant images to embed
Run the `loader.py` to embed the plant images to the ChromaDB vector store

## 👯 Use Cases

- Educational plant ID assistant.
- Smart botany guides in mobile applications.
- Herbarium digitization and classification systems.

## ✈️ Future Improvements

- Add multilingual support for plant names.
- Integrate feedback loop for retraining embeddings.

## 👷👷 Contributors

- Kirsten Dwayne Dizon – AI/ML Speacialist 
- Arron Kian Parejas – Data Scientist & AI Developer  
