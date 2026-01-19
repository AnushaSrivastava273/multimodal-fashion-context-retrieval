import json
import faiss
import numpy as np
import os
from utils import load_model, get_text_embedding

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "embeddings.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "metadata.json")

def search(query_text, top_k=5):
    # Ensure artifacts exist
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("Error: Index or metadata not found. Run index.py first.")
        return []

    # Load model and artifacts
    model, _ = load_model()
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        mapping = json.load(f)
    
    # 1. Encode query text into the CLIP latent space
    q_emb = get_text_embedding(query_text, model).astype('float32').reshape(1, -1)
    
    # 2. Search all vectors (Global + Regional Crops)
    # We retrieve 3x top_k to account for multiple vectors per image
    distances, indices = index.search(q_emb, k=top_k * 3)
    
    # 3. Aggregate Results (Max-Similarity Alignment)
    # Since each image has 3 vectors, we find the best match among them
    results = {}
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1: continue
        
        # Each image i has vectors at indices [3*i, 3*i+1, 3*i+2]
        img_id = str(idx // 3)
        path = mapping.get(img_id)
        if not path: continue
        
        # Max-pool across the global/top/bottom features
        if path not in results or score > results[path]:
            results[path] = float(score)
            
    # Sort by descending similarity and return top_k
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python query.py 'professional attire in an office'")
    else:
        user_query = " ".join(sys.argv[1:])
        print(f"Searching for: '{user_query}'")
        top_matches = search(user_query)
        
        if not top_matches:
            print("No results found.")
        for i, (path, score) in enumerate(top_matches):
            print(f"{i+1}. [{score:.4f}] {os.path.basename(path)}")
