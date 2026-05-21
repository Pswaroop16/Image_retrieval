import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import open_clip
from transformers import SiglipProcessor, SiglipModel
import clip

DATABASE_DIR = "database_images"
QUERY_IMAGE_PATH = "/home/swaroop/task1/query.jpg/query1.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME_CLIP = "ViT-B/32"
MODEL_NAME_CLIP2 = "ViT-B-32"
PRETRAINED_CLIP2 = "openai"
MODEL_NAME_SIGLIP = "google/siglip-base-patch16-224"

def load_model(model_name, preprocess_fn=None):
    """Load model and preprocessing function based on model type."""
    if model_name == 'clip':
        model, preprocess = clip.load(MODEL_NAME_CLIP, device=DEVICE)
    elif model_name == 'clip2':
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME_CLIP2, pretrained=PRETRAINED_CLIP2)
        model = model.to(DEVICE)
    else:
        model = SiglipModel.from_pretrained(MODEL_NAME_SIGLIP).to(DEVICE)
        preprocess = SiglipProcessor.from_pretrained(MODEL_NAME_SIGLIP)
    model.eval()
    return model, preprocess

def get_image_embedding(image_path, model, preprocess):
    """Get image embedding for different models."""
    image = Image.open(image_path).convert("RGB")
    input_data = preprocess(image).unsqueeze(0).to(DEVICE) if 'clip' in str(type(model)).lower() else preprocess(images=image, return_tensors="pt")["pixel_values"].to(DEVICE)
    with torch.no_grad():
        if 'clip' in str(type(model)).lower():
            embedding = model.encode_image(input_data)
        else:
            embedding = model.get_image_features(input_data)
        return embedding / embedding.norm(dim=-1, keepdim=True)

def calculate_similarity(query_embedding, image_embeddings):
    """Calculate similarity score for the query image."""
    
    query_embedding = query_embedding.cpu().numpy()
    image_embeddings = image_embeddings.cpu().numpy() if isinstance(image_embeddings, torch.Tensor) else image_embeddings
    
    return np.dot(image_embeddings, query_embedding.T).squeeze()

def process_database_images(model_type, model, preprocess):
    """Process all images in the database and extract embeddings."""
    embeddings, names = [], []
    for img_name in tqdm(os.listdir(DATABASE_DIR)):
        img_path = os.path.join(DATABASE_DIR, img_name)
        if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
            embedding = get_image_embedding(img_path, model, preprocess)
            embeddings.append(embedding.cpu().numpy()) 
            names.append(img_name)
    return np.vstack(embeddings), names

def display_result(model_name, result, image_names):
    """Display the result for the current model."""
    print(f"\n{model_name.upper()} Results:")
    print(f"  Best Match Image: {result['best_match']}")
    print(f"  Similarity Score: {result['score']:.2f}%")
    print(f"  Top 3 Matching Images (Sorted by Similarity):")
    
    top_3_indices = np.argsort(result['similarities'])[-3:][::-1]
    for idx in top_3_indices:
        print(f"    {image_names[idx]} - {result['similarities'][idx] * 100:.2f}%")

def main():
    query_embeddings = {}

    models = ['clip', 'clip2', 'siglip']
    results = {}
    
    for model_name in models:
        print(f"\n--- Running {model_name.upper()} ---")
        model, preprocess = load_model(model_name)
        image_embeddings, image_names = process_database_images(model_name, model, preprocess)
        query_embedding = get_image_embedding(QUERY_IMAGE_PATH, model, preprocess)
        similarities = calculate_similarity(query_embedding, image_embeddings)

        best_match_idx = np.argmax(similarities)
        results[model_name] = {
            "best_match": image_names[best_match_idx],
            "score": similarities[best_match_idx] * 100,
            "similarities": similarities
        }

        display_result(model_name, results[model_name], image_names)

    best_model = max(results, key=lambda x: results[x]['score'])
    best_result = results[best_model]
    
    print("\n--- BEST MATCH RESULT ---")
    print(f"Best Matched Image: {best_result['best_match']}")
    print(f"Highest score matched: {best_result['score']:.2f}%")
    print(f"Model that gave best match: {best_model.upper()}")

if __name__ == "__main__":
    main()
