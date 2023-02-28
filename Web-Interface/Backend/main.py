# Import flask for API
from flask import Flask, request, jsonify

# Import keras and other libraries for processing the MRI image
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Define the flask app
app = Flask(__name__)

# Load the model
model = load_model('model.h5')

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
	file = request.files['file']
	# Save the file to ./uploads
	filepath = './uploads/' + file.filename
	file.save(filepath)
	# Make prediction
	prediction = predict_class(filepath, model)
	# Convert the response to a string
	response = str(prediction)
	return response


if __name__ == "__main__":
	app.run(debug=True)
	
