from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)

# Load your trained model (replace 'your_model.h5' with the actual model file)
model = load_model(r'C:/Users/fbajo/OneDrive/Desktop/flower health classification/model/CNNFLOWER.h5')

# Define the target size expected by the model
target_size = (160, 160)

# Define a function to preprocess the image for model prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if the file is not empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Save the uploaded file to a temporary location
        img_path = 'temp_upload.png'
        file.save(img_path)

        # Preprocess the image for model prediction
        img_array = preprocess_image(img_path)

        # Make prediction using the loaded model
        prediction = model.predict(img_array)

        # Assuming 0 is healthy and 1 is unhealthy (you may need to adjust this based on your model output)
        if prediction[0][0] < 1:
            prediction_label = 'Healthy'
        else:
            prediction_label = 'Unhealthy'

        # You can perform further processing or return additional information here

        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, ssl_context='adhoc',port=8080)  # Use 'adhoc' for a self-signed certificate
