'''
# Note
app -> web app itself
__name__ -> parameter
'''
from flask import Flask
from flask_cors import CORS
import tensorflow as tf

app = Flask(__name__)
CORS(app) # -> make it accessible from browsers

MODEL_PATH = 'wheat_mobilenet_finetuned.h5'
model = None

def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Successfully loaded a model!")
    except Exception as e:
        print(f"Error!: {e}")

@app.route('/WheatDisease')

def WheatDisease():
    return { 'status' : 'ok', 'model_loaded': model is not None}

if __name__ == '__main__':
    load_model()
    app.run(debug=True)