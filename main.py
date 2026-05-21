import os
import numpy as np
from tqdm import tqdm
import config
import models
import utils

def process_database_images(model_name, model, preprocess):
    """Process all images in the database and extract embeddings."""
    embeddings, names = [], []
    for img_name in tqdm(os.listdir(config.DATABASE_DIR)):
        img_path = os.path.join(config.DATABASE_DIR, img_name)
        if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
            embedding = models.get_image_embedding(img_path, model, preprocess)
            embeddings.append(embedding.cpu().numpy()) 
            names.append(img_name)
    return np.vstack(embeddings), names

def main():
    models_list = ['clip', 'clip2', 'siglip']
    results = {}
    
    for model_name in models_list:
        print(f"\n--- Running {model_name.upper()} ---")
        model, preprocess = models.load_model(model_name)
        image_embeddings, image_names = process_database_images(model_name, model, preprocess)
        query_embedding = models.get_image_embedding(config.QUERY_IMAGE_PATH, model, preprocess)
        similarities = utils.calculate_similarity(query_embedding, image_embeddings)

        best_match_idx = np.argmax(similarities)
        results[model_name] = {
            "best_match": image_names[best_match_idx],
            "score": similarities[best_match_idx] * 100,
            "similarities": similarities
        }

        utils.display_result(model_name, results[model_name], image_names)

    best_model = max(results, key=lambda x: results[x]['score'])
    best_result = results[best_model]
    
    print("\n--- BEST MATCH RESULT ---")
    print(f"Best Matched Image: {best_result['best_match']}")
    print(f"Highest score matched: {best_result['score']:.2f}%")
    print(f"Model that gave best match: {best_model.upper()}")

if __name__ == "__main__":
    main()