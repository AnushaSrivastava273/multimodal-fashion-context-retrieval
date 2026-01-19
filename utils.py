import torch
import clip
from PIL import Image

# Use GPU if available for faster feature extraction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Loads the ViT-L/14 CLIP model which provides high-dimension semantic features."""
    model, preprocess = clip.load("ViT-L/14", device=DEVICE)
    return model, preprocess

@torch.no_grad()
def get_image_embeddings(image_path, model, preprocess):
    """
    Generates three embeddings per image: Global, Top-half, and Bottom-half.
    This decomposition helps the system focus on specific garments without 
    losing overall scene context.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # Define crops: Global, Top (garment proxy), Bottom (garment proxy)
    crops = [
        img,                            # Global
        img.crop((0, 0, w, h // 2)),    # Top half
        img.crop((0, h // 2, w, h))     # Bottom half
    ]
    
    embeddings = []
    for crop in crops:
        # Preprocess and encode each crop separately
        processed = preprocess(crop).unsqueeze(0).to(DEVICE)
        emb = model.encode_image(processed)
        # Normalize to unit length for Cosine Similarity during retrieval
        emb /= emb.norm(dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy().flatten())
        
    return embeddings

@torch.no_grad()
def get_text_embedding(text, model):
    """Encodes natural language queries into the same space as the images."""
    text_tokens = clip.tokenize([text]).to(DEVICE)
    text_emb = model.encode_text(text_tokens)
    text_emb /= text_emb.norm(dim=-1, keepdim=True)
    return text_emb.cpu().numpy().flatten()
