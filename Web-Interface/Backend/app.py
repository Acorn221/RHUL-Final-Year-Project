# Import flask for API
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Import keras and other libraries for processing the MRI image
from keras.models import load_model
from keras.preprocessing import image
from os import path, remove
import numpy as np
import tensorflow as tf

# Define the flask app
app = Flask(__name__)

relativePath = path.dirname(path.abspath(__file__))

# Load the model TODO: add model to test with 
#model = load_model('model.h5')

# Define the function to predict the class of the MRI image
def predict_class(img_path, model):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = x/255
	preds = model.predict(x)
	return preds

# Define the route for the API
@app.route('/predict', methods=['GET', 'POST'])
def upload():
	# Get the file from the request
	uploadedFile = request.files['file']
	# Save the file to ./uploads
	filepath = relativePath + '/uploads/' + secure_filename(uploadedFile.filename)
	uploadedFile.save(filepath)

	age = request.form['age']
	sex = request.form['sex']
	mmse = request.form['mmse']

	# Make prediction TODO: make prediction
	# prediction = predict_class(filepath, model)
	prediction = {'0': '0.0', '1': '0.82', '2': '0.18', '3': '0.01'}
	# Convert the response to a string
	response = str(prediction)

	# Delete the file from the server
	remove(filepath)

	return response


if __name__ == "__main__":
	app.run(debug=True, port=3001)
	
