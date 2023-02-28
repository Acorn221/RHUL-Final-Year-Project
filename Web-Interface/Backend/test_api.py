# This is to test the API, by uploading an image of the MRI scan and checking what it responds with
# Import the Flask testing library
from Web-Interface.Backend.main import app


def test_endpoint():
	# Create a test client using the Flask application configured for testing
	with app.test_client() as client:
			# Send a GET request to the API endpoint
			response = client.post('/predict', data=dict(file=(open('test_image.jpg', 'rb'), 'test_image.jpg')))
			# Check that the response is valid
			assert response.status_code == 200
			# Check that the response is a JSON object
			assert response.is_json
			# Check that the JSON object contains the key 'prediction'
			assert 'prediction' in response.get_json()