# 🔍 Multi-Model Image Retrieval System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991?logo=openai&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

The **Multi-Model Image Retrieval System** is an AI-powered content-based image retrieval (CBIR) pipeline that finds the most visually similar image from a database for a given query image. It leverages three state-of-the-art vision-language embedding models — **OpenAI CLIP**, **OpenCLIP**, and **Google SigLIP** — and compares their outputs to surface the best possible match.

The system computes dense visual embeddings for both the query and all database images, ranks candidates using cosine similarity, and reports the top matches along with similarity scores. By running all three models in parallel and aggregating results, it provides a reliable, model-agnostic retrieval result.

---

## Features

- 🤖 **Three Embedding Models** — Runs OpenAI CLIP, OpenCLIP (ViT-B-32), and Google SigLIP simultaneously for robust retrieval.
- 📐 **Cosine Similarity Ranking** — Normalized embeddings ensure accurate similarity measurement.
- 🏆 **Best-Model Selection** — Automatically identifies which model produced the highest-confidence match.
- 📋 **Top-3 Results per Model** — Displays the top 3 matching images with similarity scores for each model.
- ⚡ **GPU Acceleration** — Automatically uses CUDA if available; falls back to CPU.
- 🔧 **Modular Architecture** — Clean separation of config, models, utilities, and main execution logic.
- 📦 **Extensible** — Easy to add new embedding models or swap the database directory.

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.8+ |
| Deep Learning Framework | PyTorch |
| Vision-Language Models | OpenAI CLIP, OpenCLIP, Google SigLIP |
| Transformers Library | Hugging Face `transformers` |
| Image Processing | Pillow (PIL) |
| Numerical Computing | NumPy |
| Progress Display | tqdm |
| Hardware Acceleration | CUDA (optional) |

---

## Project Structure

```
image-retrieval/
│
├── config.py               # Central configuration: paths, device, model names
├── models.py               # Model loading and image embedding logic
├── utils.py                # Similarity calculation and result display helpers
├── main.py                 # Main orchestration script (entry point)
├── image_retrieval.py      # Standalone self-contained script (alternative runner)
├── requirements.txt        # Python dependencies
│
├── database_images/        # Folder containing candidate images for retrieval
│   ├── image1.jpg
│   ├── image2.png
│   └── ...
│
└── query.jpg               # Your query image (path set in config.py)
```

### Key Files

| File | Purpose |
|---|---|
| `config.py` | Single source of truth for all configurable parameters (paths, model identifiers, device) |
| `models.py` | Loads CLIP / OpenCLIP / SigLIP models and extracts normalized image embeddings |
| `utils.py` | Computes cosine similarity between query and database embeddings; formats output |
| `main.py` | Orchestrates the full retrieval pipeline across all three models |
| `image_retrieval.py` | Equivalent standalone script — useful for quick testing without the modular setup |
| `requirements.txt` | Lists all Python package dependencies |

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/image-retrieval.git
cd image-retrieval
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Additional Required Packages

The `requirements.txt` covers the core dependencies. Install the remaining packages manually:

```bash
# OpenCLIP
pip install open_clip_torch

# Hugging Face Transformers (for SigLIP)
pip install transformers

# OpenAI CLIP (installed directly from GitHub)
pip install git+https://github.com/openai/CLIP.git
```

---

## Requirements

```
torch
torchvision
numpy
pillow
tqdm
open_clip_torch
transformers
git+https://github.com/openai/CLIP.git
```

**System Requirements:**

- Python 3.8 or higher
- CUDA-compatible GPU (optional but strongly recommended for faster inference)
- At least 4 GB RAM; 8 GB+ recommended for large image databases

---

## Configuration

All configurable parameters live in `config.py`:

```python
# config.py

DATABASE_DIR = "database_images"          # Path to the folder containing candidate images
QUERY_IMAGE_PATH = "path/to/query.jpg"    # Path to your query image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # Auto-selects GPU or CPU

MODEL_NAME_CLIP    = "ViT-B/32"                          # OpenAI CLIP model variant
MODEL_NAME_CLIP2   = "ViT-B-32"                          # OpenCLIP model variant
PRETRAINED_CLIP2   = "openai"                            # OpenCLIP pretrained weights
MODEL_NAME_SIGLIP  = "google/siglip-base-patch16-224"    # SigLIP model from HuggingFace
```

**Before running**, update the following:

| Parameter | Description |
|---|---|
| `DATABASE_DIR` | Directory containing all images to search through |
| `QUERY_IMAGE_PATH` | Full path to the image you want to find matches for |

---

## Usage

### Step 1: Prepare Your Data

Place all candidate/database images inside the `database_images/` folder:

```
database_images/
├── cat1.jpg
├── dog2.png
├── landscape3.jpeg
└── ...
```

Supported formats: `.jpg`, `.jpeg`, `.png`

### Step 2: Set the Query Image

Update `QUERY_IMAGE_PATH` in `config.py` to point to your query image.

### Step 3: Run the Pipeline

```bash
python main.py
```

Or use the self-contained alternative script:

```bash
python image_retrieval.py
```

---

## Workflow / Pipeline

The system follows a straightforward inference-and-rank pipeline:

```
Query Image
     │
     ▼
┌─────────────────────────────────────────┐
│   For each model (CLIP, OpenCLIP, SigLIP)│
│                                          │
│  1. Load Model & Preprocessor            │
│  2. Encode Query Image → Query Embedding │
│  3. Encode All Database Images           │
│     → Database Embeddings Matrix         │
│  4. Compute Cosine Similarity            │
│     (Query Embedding · DB Embeddings)    │
│  5. Rank by Similarity Score             │
│  6. Return Top-3 Matches + Best Match    │
└─────────────────────────────────────────┘
     │
     ▼
Compare Best Match Score Across All 3 Models
     │
     ▼
Report Overall Best Matched Image + Model
```

**Detailed Steps:**

1. **Model Loading** (`models.py → load_model`) — Each model (CLIP, OpenCLIP, SigLIP) is loaded with its corresponding preprocessor and moved to the target device (GPU/CPU).

2. **Database Embedding Extraction** (`main.py → process_database_images`) — Every image in `database_images/` is read, preprocessed, and passed through the model's image encoder. The resulting embedding is L2-normalized to unit length.

3. **Query Embedding Extraction** (`models.py → get_image_embedding`) — The query image undergoes the same preprocessing and encoding pipeline, producing a normalized query vector.

4. **Similarity Computation** (`utils.py → calculate_similarity`) — Cosine similarity is computed as the dot product between the normalized query vector and the database embedding matrix, yielding a similarity score in `[0, 1]` for each candidate.

5. **Ranking & Aggregation** (`main.py → main`) — Candidates are ranked by similarity. The top-3 results are displayed per model, and the overall best match across all models is selected.

---

## Model Architecture

### OpenAI CLIP (`ViT-B/32`)
A Vision Transformer (ViT) with patch size 32, trained contrastively on 400M image-text pairs. Produces 512-dimensional image embeddings. Loaded via the `clip` package.

### OpenCLIP (`ViT-B-32`, OpenAI weights)
An open reproduction of CLIP with the same ViT-B/32 architecture. Loaded via the `open_clip` package with OpenAI pretrained weights, enabling reproducibility and broader fine-tuning options.

### Google SigLIP (`siglip-base-patch16-224`)
A sigmoid-loss variant of CLIP trained by Google. Uses patch size 16 on 224×224 inputs. Produces 768-dimensional embeddings. Loaded via Hugging Face `transformers`.

All models use **L2 normalization** on their output embeddings, converting dot-product similarity to cosine similarity.

---

## Example Output

```
--- Running CLIP ---
100%|████████████████████| 120/120 [00:18<00:00,  6.5it/s]

CLIP Results:
  Best Match Image: retrieval_007.jpg
  Similarity Score: 87.43%
  Top 3 Matching Images (Sorted by Similarity):
    retrieval_007.jpg - 87.43%
    retrieval_023.jpg - 81.20%
    retrieval_045.jpg - 79.55%

--- Running CLIP2 ---
...

--- Running SIGLIP ---
...

--- BEST MATCH RESULT ---
Best Matched Image: retrieval_007.jpg
Highest score matched: 89.12%
Model that gave best match: SIGLIP
```

---

## Future Improvements

- **Embedding Caching** — Persist database embeddings to disk (`.npy` or a vector database like FAISS) to avoid recomputation on every run.
- **FAISS Integration** — Use Facebook's FAISS library for approximate nearest-neighbor search, enabling retrieval from millions of images efficiently.
- **Ensemble Scoring** — Combine similarity scores from all three models via weighted averaging for a more robust final ranking.
- **Web Interface** — Add a Gradio or Streamlit UI for drag-and-drop query image upload and visual result display.
- **Batch Query Support** — Extend the pipeline to handle multiple query images in a single run.
- **Support for More Formats** — Extend image format support to `.bmp`, `.webp`, `.tiff`, etc.
- **Evaluation Metrics** — Add Precision@K, Recall@K, and mAP metrics for benchmarking retrieval accuracy on labeled datasets.
- **Docker Support** — Containerize the application for reproducible, environment-agnostic deployment.
- **Fine-tuning** — Allow domain-specific fine-tuning of the embedding models on custom datasets for improved retrieval in specialized domains.

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: clip` | CLIP not installed | Run `pip install git+https://github.com/openai/CLIP.git` |
| `ModuleNotFoundError: open_clip` | OpenCLIP not installed | Run `pip install open_clip_torch` |
| `CUDA out of memory` | GPU VRAM insufficient | Reduce database size or set `DEVICE = "cpu"` in `config.py` |
| `FileNotFoundError` on query image | Wrong path in config | Update `QUERY_IMAGE_PATH` in `config.py` to the correct absolute path |
| `No images found in database` | Wrong directory or no supported files | Verify `DATABASE_DIR` exists and contains `.jpg`/`.png`/`.jpeg` files |
| Slow inference on CPU | No GPU available | Use a CUDA-enabled machine or reduce database size |
| `PIL.UnidentifiedImageError` | Corrupt or unsupported image file | Remove or replace the problematic file in `database_images/` |

---

## License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

> **Note:** This project is for research and educational purposes. Model weights (CLIP, OpenCLIP, SigLIP) are subject to their respective upstream licenses from OpenAI, LAION, and Google.
