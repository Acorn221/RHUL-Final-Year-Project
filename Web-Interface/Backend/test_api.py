# This is to test the API, by uploading an image of the MRI scan and checking what it responds with
# Import the Flask testing library
from Backend.main import app
from os import path

# Setting the relative path to the test files
relativePath = path.dirname(path.abspath(__file__))


endpointExt = "/predict"

"""
This test is to check if the API responds with a 200 status code when a valid image is uploaded
"""
def test_endpoint():
	# Create a test client using the Flask application configured for testing
	with app.test_client() as client:
			# Send a GET request to the API endpoint
			response = client.post(endpointExt, data=dict(file=(open(relativePath+'Target.png', 'rb'), 'test_image.jpg')))
			# Check that the response is valid
			assert response.status_code == 200
			# Check that the response is a JSON object
			assert response.is_json
			# Check that the JSON object contains the key 'prediction'
			assert 'prediction' in response.get_json()

"""
This test is to check if the API responds with a 400 status code when an invalid image is uploaded
"""
def test_not_image():
	# Create a test client using the Flask application configured for testing
	with app.test_client() as client:
			# Send a GET request to the API endpoint
			response = client.post(endpointExt, data=dict(file=(open(relativePath+'Not_An_Image.exe', 'rb'), 'Not_An_Image.exe')))
			# Check that the response is valid
			assert response.status_code == 400

"""
This test is to check if the API responds with a 400 status code when a corrupted image is uploaded
"""
def test_corrupted_image():
	# Create a test client using the Flask application configured for testing
	with app.test_client() as client:
			# Send a GET request to the API endpoint
			response = client.post(endpointExt, data=dict(file=(open(relativePath+'corruptedImage.png', 'rb'), 'corruptedImage.png')))
			# Check that the response is valid
			assert response.status_code == 400

"""
This test is to check if the API responds with a 200 status code when a valid image is uploaded
It also checks if the response is a JSON object and if it contains the key 'prediction'
"""
def test_mri_image():
	# Create a test client using the Flask application configured for testing
	with app.test_client() as client:
			# Send a GET request to the API endpoint
			response = client.post(endpointExt, data=dict(file=(open(relativePath+'MRI_Scan.png', 'rb'), 'MRI_Scan.png')))
			# Check that the response is valid
			assert response.status_code == 200
			# Check that the response is a JSON object
			assert response.is_json
			# Check that the JSON object contains the key 'prediction'
			assert 'prediction' in response.get_json()
