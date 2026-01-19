# Multimodal Fashion & Context Retrieval

A CLIP-based region-aware image retrieval system for fashion queries involving
clothing attributes, colors, and scene context.

---

## Problem Statement

The goal of this project is to build an intelligent multimodal search engine that
retrieves fashion images based on natural language descriptions. The system is
designed to understand not only *what* a person is wearing, but also *where* they
are and the overall *vibe* or context of the scene.

A key challenge in fashion retrieval is handling **compositional queries** such as:
"a red shirt with blue pants".

Standard global embedding models like CLIP often fail at this because they treat
the image as a single “bag of visual concepts”. This project addresses that
limitation.

---

## Key Idea: Region-Aware Semantic Indexing (RASI)

This system uses a **multi-vector representation per image** instead of a single
global embedding.

Each image is indexed using:
- **Global embedding**: captures scene, environment, and overall style
  (e.g., office, park, formal, casual)
- **Regional embeddings**: capture localized garment attributes such as
  color and clothing type (approximated using top and bottom image regions)

By decoupling scene context from garment-level details, the system can correctly
handle fine-grained and compositional fashion queries.

---

## Architecture Overview

### 1. Offline Indexing (Part A)
- Images are processed using CLIP (ViT-L/14)
- For each image:
  - One global embedding is extracted from the full image
  - Two regional embeddings are extracted from the top and bottom halves
- All embeddings are stored in a FAISS index for efficient similarity search

### 2. Online Retrieval (Part B)
- A natural language query is encoded using CLIP
- Retrieval is performed in two stages:
  1. Global context matching (scene and style)
  2. Region-level alignment for garment attributes
- Results are ranked by aggregating region matches per image

This approach improves over vanilla CLIP for queries involving multiple attributes.

---

## Repository Structure

.
├── data/
│   └── images/          # Fashion images (dataset)
├── utils.py             # CLIP loading and encoding utilities
├── index.py             # Offline indexing pipeline
├── query.py             # Query-time retrieval script
├── requirements.txt     # Python dependencies
└── README.md

---

## Dataset

The system is designed to work with **500–1,000 fashion images** containing
variation across:
- Environments: office, street, park, home
- Clothing types: formal, casual, outerwear
- Colors: diverse garment color palette

For prototyping and validation, a representative subset of images can be used.
The architecture scales without modification to larger datasets.

---

## How to Run

### 1. Install Dependencies

pip install -r requirements.txt

### 2. Build the Index

python index.py

This extracts global and regional embeddings and builds the FAISS index.

### 3. Run a Query

python query.py "a red tie and a white shirt in a formal office"

The script returns the top-k matching image paths.

---

## Example Queries Supported

- "A person in a bright yellow raincoat"
- "Professional business attire inside a modern office"
- "Someone wearing a blue shirt sitting on a park bench"
- "Casual weekend outfit for a city walk"
- "A red tie and a white shirt in a formal setting"

---

## Scalability & Zero-Shot Capability

- Uses FAISS for efficient nearest-neighbor search
- Scales linearly with dataset size (multi-vector per image)
- No task-specific training required
- Leverages CLIP’s zero-shot generalization for unseen descriptions

---

## Future Work

- **Location & Weather Awareness**:
  Integrate scene classifiers (e.g., Places365) and weather cues as metadata
- **Precision Improvements**:
  Add a cross-encoder re-ranking stage for complex spatial relationships
- **Learning from Feedback**:
  Tune similarity weights using user interaction data

---

## Notes

This project prioritizes **ML reasoning and representation design** over
infrastructure-heavy solutions, in line with the assignment requirements.
