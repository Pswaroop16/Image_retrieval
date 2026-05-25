# 🔍 Image Similarity Search

A Python-based image similarity search pipeline that compares a query image against a database of images using three state-of-the-art vision-language embedding models: **CLIP**, **OpenCLIP**, and **SigLIP**.

---

## 📌 Overview

This project extracts image embeddings using multiple pretrained models and computes cosine similarity to find the best matching image from a local database. It then reports the top matches per model and identifies which model gave the most confident result.

---

## 🗂️ Project Structure

```
├── main.py           # Entry point — orchestrates the search pipeline
├── models.py         # Model loading and image embedding extraction
├── utils.py          # Similarity calculation and result display
├── config.py         # Paths, device, and model configuration
├── requirements.txt  # Python dependencies
└── database_images/  # Folder containing images to search against
```

---

## 🧠 Models Used

| Model Key | Architecture | Source |
|-----------|-------------|--------|
| `clip`    | ViT-B/32    | OpenAI CLIP |
| `clip2`   | ViT-B-32    | OpenCLIP (openai pretrained) |
| `siglip`  | ViT-B patch16-224 | Google SigLIP (via HuggingFace) |

---

## ⚙️ Configuration

Edit `config.py` to set your paths and model preferences:

```python
DATABASE_DIR = "database_images"          # Directory with images to search
QUERY_IMAGE_PATH = "path/to/query.jpg"    # Query image path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME_CLIP   = "ViT-B/32"
MODEL_NAME_CLIP2  = "ViT-B-32"
PRETRAINED_CLIP2  = "openai"
MODEL_NAME_SIGLIP = "google/siglip-base-patch16-224"
```

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/image-similarity-search.git
cd image-similarity-search
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** CLIP is installed directly from GitHub. Ensure you have `git` available.

### 4. Prepare your images

- Place all database images (`.jpg`, `.jpeg`, `.png`) into the `database_images/` folder.
- Set your query image path in `config.py`.

---

## ▶️ Usage

```bash
python main.py
```

### Example Output

```
--- Running CLIP ---
CLIP Results:
  Best Match Image: image_042.jpg
  Similarity Score: 87.43%
  Top 3 Matching Images (Sorted by Similarity):
    image_042.jpg - 87.43%
    image_017.jpg - 81.20%
    image_033.jpg - 78.95%

--- Running CLIP2 ---
...

--- BEST MATCH RESULT ---
Best Matched Image: image_042.jpg
Highest score matched: 89.12%
Model that gave best match: SIGLIP
```

---

## 📦 Requirements

```
torch
torchvision
numpy
pillow
tqdm
git+https://github.com/openai/CLIP.git
open_clip_torch
transformers
```

> Install all dependencies via `pip install -r requirements.txt`.

---

## 🛠️ How It Works

1. **Load Models** — Each of the three models (CLIP, OpenCLIP, SigLIP) is loaded with its corresponding preprocessor.
2. **Extract Database Embeddings** — All images in `database_images/` are encoded into normalized embedding vectors.
3. **Extract Query Embedding** — The query image is encoded using the same model.
4. **Compute Similarity** — Cosine similarity is computed between the query and all database embeddings.
5. **Report Results** — Top matches are printed per model, and the overall best match across all models is reported.

---

## 💡 Notes

- GPU is used automatically if CUDA is available; falls back to CPU otherwise.
- Embeddings are L2-normalized before similarity computation (cosine similarity).
- Similarity scores are displayed as percentages (0–100%).

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
