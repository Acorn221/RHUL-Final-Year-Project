import AutomatedTestingLibrary.Classes.AutomatedTesting as t
import AutomatedTestingLibrary.Classes.Model as m
from utils import MobileNetModel
import json

model = m.Model(MobileNetModel, "MobileNetV2")

automatedTesting = t.AutomatedTesting([model], "Alzheimers-classification/data", "output")

def test_load():
	# Load the model
	model.loadModel()
	# Check if the model is loaded
	assert model.sequentialModel is not None

def test_train():
	# Load the model
	model.loadModel()

	# compile the model
	model.compileModel(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# load the training data
	automatedTesting.loadTrainingData()

	# Train the model
	model.fit(automatedTesting.train_ds, automatedTesting.test_ds, epochs=1, steps_per_epoch=1)

	# Check if the model is trained
	assert model.sequentialModel.predict(automatedTesting.test_ds.next()[0]) is not None

# Checking if the model history is saved
def test_save():
	# Save the model
	model.saveHistory()

	# Check if the JSON model history is saved

	f = open(m.modelDir+model.modelName+'.json', "r")

	assert json.load(f) is not None