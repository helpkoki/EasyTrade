from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import os
import requests

app = Flask(__name__)
CORS(app)

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet", include_top=True)

# eBay API configuration (you need to replace this with a valid token)
EBAY_APP_ID = "YOUR_EBAY_APP_ID"
EBAY_API_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"

@app.route("/")
def home():
    return "Image Classification Price Estimator API is running."

@app.route("/predict_price", methods=["POST"])
def predict_price():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    img_file = request.files["image"]
    img_path = os.path.join("uploads", img_file.filename)
    img_file.save(img_path)

    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict using MobileNetV2
        predictions = model.predict(x)
        decoded_predictions = decode_predictions(predictions, top=1)[0]
        object_class = decoded_predictions[0][1]

        # Prepare eBay API request
        headers = {
            "Authorization": f"Bearer {EBAY_APP_ID}",
            "Content-Type": "application/json",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US"
        }

        params = {
            "q": object_class,
            "limit": 5
        }

        response = requests.get(EBAY_API_URL, headers=headers, params=params)

        if response.status_code == 200:
            ebay_data = response.json()
            items = ebay_data.get("itemSummaries", [])

            prices = []
            for item in items:
                price = item.get("price", {}).get("value")
                if price:
                    prices.append(float(price))

            avg_price = sum(prices) / len(prices) if prices else 0

            os.remove(img_path)

            return jsonify({
                "detected_object": object_class,
                "predicted_average_price": f"${avg_price:.2f}",
                "sample_items": items
            })
        else:
            return jsonify({
                "detected_object": object_class,
                "error": "Could not retrieve pricing data from eBay",
                "ebay_status": response.status_code
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, port=5000)
