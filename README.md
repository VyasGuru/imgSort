# imgSort

This is a Flask-based web application that allows users to upload multiple images, generate image embeddings using the CLIP model (`openai/clip-vit-base-patch32`), and cluster the images into folders based on visual similarity. Users can then download the clustered images as a ZIP archive.

## Features

- Upload one or more image files (PNG, JPG, JPEG, WEBP)
- Generate embeddings using OpenAI's CLIP model
- Cluster images using KMeans (with 4 clusters by default)
- Download the sorted image clusters as a ZIP file

## Tech Stack

- **Backend**: Flask (Python)
- **Model**: CLIP (`openai/clip-vit-base-patch32`) via Hugging Face Transformers
- **Clustering**: KMeans (from `scikit-learn`)
- **Image Handling**: PIL (Pillow), NumPy
- **Frontend**: HTML (via Flask templates)
- **Deployment Ready**: Easily hostable on any Python server with CUDA or CPU fallback

## Directory Structure

.
├── app.py # Main Flask application
├── templates/
│ └── index.html # Frontend upload page
├── uploads/ # Temporary image upload directory
├── clusters/ # Output clusters (auto-generated)




