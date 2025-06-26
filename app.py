from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
from transformers import CLIPProcessor, CLIPModel
from sklearn.cluster import KMeans
from PIL import Image
import torch
import os, shutil, numpy as np, io, zipfile

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
CLUSTER_FOLDER = "clusters"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLUSTER_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file:
        return jsonify({'success': False, 'message': 'No file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return jsonify({'success': True, 'filename': filename})

@app.route('/generate_embeddings', methods=['POST'])
def generate_embeddings():
    images, filenames = [], []
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            img = Image.open(os.path.join(UPLOAD_FOLDER, fname)).convert("RGB")
            images.append(img)
            filenames.append(fname)

    if not images:
        return jsonify({'success': False, 'message': 'No valid images uploaded'}), 400

    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        np_feats = features.cpu().numpy()

    kmeans = KMeans(n_clusters=4, random_state=42).fit(np_feats)
    labels = kmeans.labels_

    # clear and rebuild cluster dirs
    shutil.rmtree(CLUSTER_FOLDER, ignore_errors=True)
    os.makedirs(CLUSTER_FOLDER)

    for i in range(4):
        os.makedirs(os.path.join(CLUSTER_FOLDER, f"cluster_{i}"))

    for fname, label in zip(filenames, labels):
        src = os.path.join(UPLOAD_FOLDER, fname)
        dst = os.path.join(CLUSTER_FOLDER, f"cluster_{label}", fname)
        shutil.copy(src, dst)

    return jsonify({'success': True})

@app.route('/download_clusters')
def download_clusters():
    zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_stream, 'w') as zipf:
        for root, _, files in os.walk(CLUSTER_FOLDER):
            for file in files:
                path = os.path.join(root, file)
                arcname = os.path.relpath(path, start=CLUSTER_FOLDER)
                zipf.write(path, arcname)
    zip_stream.seek(0)
    return send_file(zip_stream, download_name="clusters.zip", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
