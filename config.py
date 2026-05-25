import os
import torch

DATABASE_DIR = "database_images"
QUERY_IMAGE_PATH = "/home/s/task/query.jpg/query1.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME_CLIP = "ViT-B/32"
MODEL_NAME_CLIP2 = "ViT-B-32"
PRETRAINED_CLIP2 = "openai"
MODEL_NAME_SIGLIP = "google/siglip-base-patch16-224"
