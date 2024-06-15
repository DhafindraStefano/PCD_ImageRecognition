from flask import Flask, request, jsonify, render_template
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification

app = Flask(__name__)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('BiscuitClassifier.pth', map_location=device)
model.eval()

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define class labels and product information
class_labels = [
    "Cadburry", "Delfi", "Malkist Roma Abon", "Malkist Roma Belgian Chocolate",
    "Malkist Roma Biskuit Kelapa", "Malkist Roma Biskuit Kelapa Kopyor", "Malkist Roma Cokelat Kelapa",
    "Malkist Roma Crackers", "Malkist Roma Keju Manis", "Malkist Roma Sandwich Chocolate",
    "Malkist Roma Sandwich Peanut Butter", "Malkist Sari Gandum", "Oreo Chocolate", "Oreo Fizzy",
    "Oreo Ice Cream", "Oreo Original", "Oreo Red Velvet", "Silverqueen", "Tango Royal Chocolate",
    "Tango Sassy Strawberry", "Tango Vanilla Delight", "Toblerone"
]


products = [
    {'nama': 'Cadburry', 'deskripsi': 'Cokelat susu yang creamy dan lezat.', 'harga': 27000, 'berat': 100, 'kandungan_kkal': 500},
    {'nama': 'Delfi', 'deskripsi': 'Cokelat dengan rasa lezat dan creamy.', 'harga': 20000, 'berat': 100, 'kandungan_kkal': 510},
    {'nama': 'Malkist Roma Abon', 'deskripsi': 'Crackers dengan taburan abon sapi yang gurih.', 'harga': 17000, 'berat': 140, 'kandungan_kkal': 510},
    {'nama': 'Malkist Roma Belgian Chocolate', 'deskripsi': 'Crackers dengan lapisan cokelat Belgia yang lezat.', 'harga': 18000, 'berat': 125, 'kandungan_kkal': 540},
    {'nama': 'Malkist Roma Biskuit Kelapa', 'deskripsi': 'Biskuit dengan rasa kelapa yang menggoda.', 'harga': 15000, 'berat': 150, 'kandungan_kkal': 480},
    {'nama': 'Malkist Roma Biskuit Kelapa Kopyor', 'deskripsi': 'Biskuit dengan cita rasa kelapa kopyor.', 'harga': 16000, 'berat': 155, 'kandungan_kkal': 490},
    {'nama': 'Malkist Roma Cokelat Kelapa', 'deskripsi': 'Crackers dengan rasa cokelat kelapa yang unik.', 'harga': 18000, 'berat': 135, 'kandungan_kkal': 530},
    {'nama': 'Malkist Roma Crackers', 'deskripsi': 'Crackers gurih dengan taburan gula yang renyah.', 'harga': 15000, 'berat': 150, 'kandungan_kkal': 480},
    {'nama': 'Malkist Roma Keju Manis', 'deskripsi': 'Crackers dengan rasa keju manis yang menggugah selera.', 'harga': 16000, 'berat': 130, 'kandungan_kkal': 520},
    {'nama': 'Malkist Roma Sandwich Chocolate', 'deskripsi': 'Sandwich crackers dengan isian cokelat manis.', 'harga': 19000, 'berat': 140, 'kandungan_kkal': 550},
    {'nama': 'Malkist Roma Sandwich Peanut Butter', 'deskripsi': 'Sandwich crackers dengan isian selai kacang yang kaya.', 'harga': 20000, 'berat': 145, 'kandungan_kkal': 560},
    {'nama': 'Malkist Sari Gandum', 'deskripsi': 'Biskuit gandum dengan rasa alami dan sehat.', 'harga': 14000, 'berat': 160, 'kandungan_kkal': 460},
    {'nama': 'Oreo Chocolate', 'deskripsi': 'Oreo dengan rasa cokelat yang lebih intens.', 'harga': 13000, 'berat': 100, 'kandungan_kkal': 480},
    {'nama': 'Oreo Fizzy', 'deskripsi': 'Oreo dengan rasa soda yang unik.', 'harga': 15000, 'berat': 100, 'kandungan_kkal': 500},
    {'nama': 'Oreo Ice Cream', 'deskripsi': 'Oreo dengan rasa es krim yang menyegarkan.', 'harga': 14000, 'berat': 100, 'kandungan_kkal': 490},
    {'nama': 'Oreo Original', 'deskripsi': 'Biskuit hitam dengan krim putih klasik.', 'harga': 12000, 'berat': 100, 'kandungan_kkal': 470},
    {'nama': 'Oreo Red Velvet', 'deskripsi': 'Oreo dengan rasa red velvet dan krim keju.', 'harga': 15000, 'berat': 100, 'kandungan_kkal': 500},
    {'nama': 'Silverqueen', 'deskripsi': 'Cokelat kacang klasik dengan rasa yang istimewa.', 'harga': 25000, 'berat': 100, 'kandungan_kkal': 520},
    {'nama': 'Tango Royal Chocolate', 'deskripsi': 'Wafer dengan lapisan cokelat royal.', 'harga': 13000, 'berat': 125, 'kandungan_kkal': 460},
    {'nama': 'Tango Sassy Strawberry', 'deskripsi': 'Wafer dengan rasa stroberi yang segar.', 'harga': 12000, 'berat': 125, 'kandungan_kkal': 450},
    {'nama': 'Tango Vanilla Delight', 'deskripsi': 'Wafer dengan rasa vanilla yang nikmat.', 'harga': 12000, 'berat': 125, 'kandungan_kkal': 450},
    {'nama': 'Toblerone', 'deskripsi': 'Cokelat Swiss dengan nougat dan almond yang ikonik.', 'harga': 30000, 'berat': 100, 'kandungan_kkal': 530}
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        image = Image.open(file.stream).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image).logits
            prediction = torch.argmax(outputs, dim=1).item()
            predicted_label = class_labels[prediction]
            product_info = next((product for product in products if product['nama'] == predicted_label), None)
            if product_info is None:
                return jsonify({'error': 'Product not found'}), 404
            return jsonify(product_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
