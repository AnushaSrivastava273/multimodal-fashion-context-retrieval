import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from utils import load_model, get_image_embeddings

# Use absolute paths where possible, or relative to the workspace root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
INDEX_SAVE_PATH = os.path.join(BASE_DIR, "embeddings.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

def build_index():
    model, preprocess = load_model()
    
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"Created {IMAGE_DIR}. Please add images and run again.")
        return

    image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}.")
        return

    # 768 is the embedding dimension for CLIP ViT-L/14
    # IndexFlatIP uses Inner Product (Cosine Similarity on normalized vectors)
    index = faiss.IndexFlatIP(768)
    mapping = {} # Store file paths to resolve after search
    
    print(f"Indexing {len(image_paths)} images...")
    
    for i, path in enumerate(tqdm(image_paths)):
        try:
            # Each image generates 3 vectors: Global, Top, Bottom
            embs = get_image_embeddings(path, model, preprocess)
            index.add(np.array(embs).astype('float32'))
            
            # Map the block of 3 vector IDs back to this specific image path
            mapping[i] = path
        except Exception as e:
            print(f"Skipping {path}: {e}")

    # Persist the index and path mapping to disk
    faiss.write_index(index, INDEX_SAVE_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(mapping, f)
    
    print(f"Done! Index saved to {INDEX_SAVE_PATH}")

if __name__ == "__main__":
    build_index()
