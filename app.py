from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your trained model
model = load_model('Trained Models/ImageNet_InceptionV3_bengali_digits.h5')

# Function to process and predict image
def prepare_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize based on your model input size
    img = np.array(img) / 255.0   # Normalize if needed
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Predict the class of the image
    image = prepare_image(file)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)

    return jsonify({"prediction": str(predicted_class[0])})

if __name__ == '__main__':
    app.run(debug=True)
