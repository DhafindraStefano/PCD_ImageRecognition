from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load your trained model
model = tf.keras.models.load_model("trained_model.h5")  # Make sure to update the path to your .h5 file

# Define the class labels (assuming you have 21 classes)
class_labels = [
    "Malkist Roma Crackers", "Malkist Roma Belgian Chocolate", "Malkist Roma Keju Manis", 
    "Malkist Roma Abon", "Malkist Roma Cokelat Kelapa", "Malkist Roma Sandwich Chocolate", 
    "Malkist Roma Sandwich Peanut Butter", "Malkist Roma Biskuit Kelapa", 
    "Malkist Roma Biskuit Kelapa Kopyor", "Malkist Sari Gandum", "Oreo Original", 
    "Oreo Chocolate", "Oreo Ice Cream", "Oreo Red Velvet", "Oreo Fizzy", 
    "Tango Vanilla Delight", "Tango Royal Chocolate", "Tango Sassy Strawberry", 
    "Silverqueen", "Toblerone", "Delfi", "Cadburry"
]

# Product information
products = [
    {
        "nama": "Malkist Roma Crackers",
        "deskripsi": "Crackers gurih dengan taburan gula yang renyah.",
        "harga": 15000,
        "berat": 150,
        "kandungan_kkal": 480
    },
    {
        "nama": "Malkist Roma Belgian Chocolate",
        "deskripsi": "Crackers dengan lapisan cokelat Belgia yang lezat.",
        "harga": 18000,
        "berat": 125,
        "kandungan_kkal": 540
    },
    {
        "nama": "Malkist Roma Keju Manis",
        "deskripsi": "Crackers dengan rasa keju manis yang menggugah selera.",
        "harga": 16000,
        "berat": 130,
        "kandungan_kkal": 520
    },
    {
        "nama": "Malkist Roma Abon",
        "deskripsi": "Crackers dengan taburan abon sapi yang gurih.",
        "harga": 17000,
        "berat": 140,
        "kandungan_kkal": 510
    },
    {
        "nama": "Malkist Roma Cokelat Kelapa",
        "deskripsi": "Crackers dengan rasa cokelat kelapa yang unik.",
        "harga": 18000,
        "berat": 135,
        "kandungan_kkal": 530
    },
    {
        "nama": "Malkist Roma Sandwich Chocolate",
        "deskripsi": "Sandwich crackers dengan isian cokelat manis.",
        "harga": 19000,
        "berat": 140,
        "kandungan_kkal": 550
    },
    {
        "nama": "Malkist Roma Sandwich Peanut Butter",
        "deskripsi": "Sandwich crackers dengan isian selai kacang yang kaya.",
        "harga": 20000,
        "berat": 145,
        "kandungan_kkal": 560
    },
    {
        "nama": "Malkist Roma Biskuit Kelapa",
        "deskripsi": "Biskuit dengan rasa kelapa yang menggoda.",
        "harga": 15000,
        "berat": 150,
        "kandungan_kkal": 480
    },
    {
        "nama": "Malkist Roma Biskuit Kelapa Kopyor",
        "deskripsi": "Biskuit dengan cita rasa kelapa kopyor.",
        "harga": 16000,
        "berat": 155,
        "kandungan_kkal": 490
    },
    {
        "nama": "Malkist Sari Gandum",
        "deskripsi": "Biskuit gandum dengan rasa alami dan sehat.",
        "harga": 14000,
        "berat": 160,
        "kandungan_kkal": 460
    },
    {
        "nama": "Oreo Original",
        "deskripsi": "Biskuit hitam dengan krim putih klasik.",
        "harga": 12000,
        "berat": 100,
        "kandungan_kkal": 470
    },
    {
        "nama": "Oreo Chocolate",
        "deskripsi": "Oreo dengan rasa cokelat yang lebih intens.",
        "harga": 13000,
        "berat": 100,
        "kandungan_kkal": 480
    },
    {
        "nama": "Oreo Ice Cream",
        "deskripsi": "Oreo dengan rasa es krim yang menyegarkan.",
        "harga": 14000,
        "berat": 100,
        "kandungan_kkal": 490
    },
    {
        "nama": "Oreo Red Velvet",
        "deskripsi": "Oreo dengan rasa red velvet dan krim keju.",
        "harga": 15000,
        "berat": 100,
        "kandungan_kkal": 500
    },
    {
        "nama": "Oreo Fizzy",
        "deskripsi": "Oreo dengan rasa soda yang unik.",
        "harga": 15000,
        "berat": 100,
        "kandungan_kkal": 500
    },
    {
        "nama": "Tango Vanilla Delight",
        "deskripsi": "Wafer dengan rasa vanilla yang nikmat.",
        "harga": 12000,
        "berat": 125,
        "kandungan_kkal": 450
    },
    {
        "nama": "Tango Royal Chocolate",
        "deskripsi": "Wafer dengan lapisan cokelat royal.",
        "harga": 13000,
        "berat": 125,
        "kandungan_kkal": 460
    },
    {
        "nama": "Tango Sassy Strawberry",
        "deskripsi": "Wafer dengan rasa stroberi yang segar.",
        "harga": 12000,
        "berat": 125,
        "kandungan_kkal": 450
    },
    {
        "nama": "Silverqueen",
        "deskripsi": "Cokelat kacang klasik dengan rasa yang istimewa.",
        "harga": 25000,
        "berat": 100,
        "kandungan_kkal": 520
    },
    {
        "nama": "Toblerone",
        "deskripsi": "Cokelat Swiss dengan nougat dan almond yang ikonik.",
        "harga": 30000,
        "berat": 100,
        "kandungan_kkal": 530
    },
    {
        "nama": "Delfi",
        "deskripsi": "Cokelat dengan rasa lezat dan creamy.",
        "harga": 20000,
        "berat": 100,
        "kandungan_kkal": 510
    },
    {
        "nama": "Cadburry",
        "deskripsi": "Cokelat susu yang creamy dan lezat.",
        "harga": 27000,
        "berat": 100,
        "kandungan_kkal": 500
    }
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file to a temporary location
    file_path = os.path.join("temp", file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)
    
    try:
        # Preprocess the image
        img = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict the class
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        # Get the product information using the predicted class index
        product_name = class_labels[predicted_class]
        product_info = next((item for item in products if item["nama"] == product_name), None)
        
        if product_info is None:
            return jsonify({'error': 'Product not found'}), 404
        
        return jsonify(product_info)
    
    finally:
        # Remove the temporary file
        os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
