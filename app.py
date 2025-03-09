from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model_cnn.h5")

# Preprocessing gambar
def preprocess_image(image):
    image = image.resize((28, 28)).convert("L")  # Ubah ke grayscale
    image = np.array(image) / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambah batch dimensi
    image = np.expand_dims(image, axis=-1)  # Tambah channel dimensi
    return image

# Route untuk halaman utama (frontend)
@app.route("/")
def home():
    return render_template("index.html")

# Endpoint untuk prediksi
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))
    image = preprocess_image(image)

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    
    return jsonify({"prediction": int(predicted_class)})

if __name__ == "__main__":
    app.run(debug=True)
