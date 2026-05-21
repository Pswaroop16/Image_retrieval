import torch
from PIL import Image
import clip
import open_clip
from transformers import SiglipProcessor, SiglipModel
import config

def load_model(model_name):
    """Load model and preprocessing function based on model type."""
    if model_name == 'clip':
        model, preprocess = clip.load(config.MODEL_NAME_CLIP, device=config.DEVICE)
    elif model_name == 'clip2':
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.MODEL_NAME_CLIP2, 
            pretrained=config.PRETRAINED_CLIP2
        )
        model = model.to(config.DEVICE)
    else:
        model = SiglipModel.from_pretrained(config.MODEL_NAME_SIGLIP).to(config.DEVICE)
        preprocess = SiglipProcessor.from_pretrained(config.MODEL_NAME_SIGLIP)
    model.eval()
    return model, preprocess

def get_image_embedding(image_path, model, preprocess):
    """Get image embedding for different models."""
    image = Image.open(image_path).convert("RGB")
    
    if 'clip' in str(type(model)).lower():
        input_data = preprocess(image).unsqueeze(0).to(config.DEVICE)
    else:
        input_data = preprocess(images=image, return_tensors="pt")["pixel_values"].to(config.DEVICE)
        
    with torch.no_grad():
        if 'clip' in str(type(model)).lower():
            embedding = model.encode_image(input_data)
        else:
            embedding = model.get_image_features(input_data)
        return embedding / embedding.norm(dim=-1, keepdim=True)