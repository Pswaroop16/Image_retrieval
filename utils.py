import numpy as np

def calculate_similarity(query_embedding, image_embeddings):
    """Calculate similarity score for the query image."""
    query_embedding = query_embedding.cpu().numpy()
    image_embeddings = image_embeddings.cpu().numpy() if isinstance(image_embeddings, float) or isinstance(image_embeddings, np.ndarray) is False else image_embeddings
    
    # Keeping the precise logic to handle torch tensors vs raw array conversion safely
    if not isinstance(image_embeddings, np.ndarray):
        image_embeddings = image_embeddings.cpu().numpy()
        
    return np.dot(image_embeddings, query_embedding.T).squeeze()

def display_result(model_name, result, image_names):
    """Display the result for the current model."""
    print(f"\n{model_name.upper()} Results:")
    print(f"  Best Match Image: {result['best_match']}")
    print(f"  Similarity Score: {result['score']:.2f}%")
    print(f"  Top 3 Matching Images (Sorted by Similarity):")
    
    top_3_indices = np.argsort(result['similarities'])[-3:][::-1]
    for idx in top_3_indices:
        print(f"    {image_names[idx]} - {result['similarities'][idx] * 100:.2f}%")