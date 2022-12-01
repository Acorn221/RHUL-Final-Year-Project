import AutomatedTestingLibrary.Classes.AutomatedTesting as t
import AutomatedTestingLibrary.Classes.Model as m
from utils import MobileNetModel

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
	model.fit(automatedTesting.train_ds, automatedTesting.test_ds, 1)

	# Check if the model is trained
	assert model.model.predict(automatedTesting.test_ds.next()[0][0].shape) is not None