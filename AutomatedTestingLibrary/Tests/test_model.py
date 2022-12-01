import AutomatedTestingLibrary.Classes.AutomatedTesting as t
import AutomatedTestingLibrary.Classes.Model as m
from utils import MobileNetModel

model = m.Model(MobileNetModel, "MobileNetV2")

def test_load():
	# Load the model
	model.loadModel()
	# Check if the model is loaded
	assert model.model is not None

def test_train():
	# Load the model
	model.loadModel()
	# Train the model
	model.trainModel()
	# Check if the model is trained
	assert model.model.predict()