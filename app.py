'''
# Note
app -> web app itself
__name__ -> parameter
'''
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app) # -> make it accessible from browsers

MODEL_PATH = 'wheat_model_v2.keras'
model = None

CLASS_NAMES = [
    "Healthy", "Mildew", "Pests", "Rust", 
    "Fusarium_Blast", "Smut-Rot", "Blight_Spot"
]

# Function to load model
def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Successfully loaded a model!")
    except Exception as e:
        print(f"Error!: {e}")

# Image Processing → resize, rgb, 
def process_img(image):
    # Resize and image and convert it to numpy array
    img = image.resize((160, 160))
    img_array = np.array(img)

    # RGB - if image doesn't have 3 channel, make it three
    if img_array.shape[-1] != 4:
        img_array = img_array[:,:, :3]
    
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis = 0)

    return img_array
# # Test image preprocessing
# test = Image.open("test_img.png")
# processed = process_img(test)
# print(processed.shape)

@app.route('/WheatDisease')
def WheatDisease():
    return { 'status' : 'ok', 'model_loaded': model is not None}

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'No model loaded.'}), 500
    else:
        try:
            # Make sure is users upload image successfully
            if 'image' not in request.files:    # request.files -> store files sent from the front-end. Has a structure like dictionary
                return jsonify({'error': 'No image file found.'}), 400
            
            # Obtain image from request.files and read it as PIL image
            file = request.files['image']
            image = Image.open(io.BytesIO(file.read()))

            # Preprocess the image data
            processedImg = process_img(image)

            # Predict
            pred = model.predict(processedImg)

            pred_classIndex = np.argmax(pred[0])
            confidence = float(pred[0][pred_classIndex])

            return jsonify({
                # 'Image':
                'Prediction' : pred
            })
        
        except Exception as e:
            return jsonify({'error': f'Prediction Error: {str(e)}'}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host = '0.0.0.0', port = 5000)