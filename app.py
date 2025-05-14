import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet", include_top=True)

@app.route("/")
def home():
    return "Image Recognition API is running."

@app.route("/recognize_image", methods=["POST"])
def recognize_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    img_file = request.files["image"]

    try:
        # Read image
        img = Image.open(BytesIO(img_file.read()))
        img = img.convert("RGB")
        img = img.resize((224, 224))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict using MobileNetV2
        predictions = model.predict(x)
        decoded_predictions = decode_predictions(predictions, top=3)[0]  # top 3 predictions

        results = [
            {"label": label, "description": desc, "probability": float(prob)}
            for (label, desc, prob) in decoded_predictions
        ]

        return jsonify({"predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
# Note: this code works on railway   
