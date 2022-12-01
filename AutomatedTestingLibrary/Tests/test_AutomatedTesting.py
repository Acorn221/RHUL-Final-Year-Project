import AutomatedTestingLibrary.Classes.AutomatedTesting as t
import AutomatedTestingLibrary.Classes.Model as m
from tensorflow import keras

from utils import MobileNetModel

models = [m.Model(MobileNetModel, "MobileNetV2")]

automatedTesting = t.AutomatedTesting(models, "Alzheimers-classification/data", "output")

def test_load_training_data():
	# Load the training data
	automatedTesting.loadTrainingData()
	# Check if the training data is loaded
	assert automatedTesting.train_ds is not None
	assert automatedTesting.test_ds is not None

def test_start():
	# Start the automated testing
	automatedTesting.start()
	# Check if the automated testing is done
	assert automatedTesting.done is True

def test_training_args():
	trainingArgs = [
		{
			"train": {
				"epochs": 5,
			},
			"compile": {
				"optimizer": keras.optimizers.Adam(learning_rate=0.001/(x * 10)),
				"loss": keras.losses.CategoricalCrossentropy(),
				"metrics": [keras.metrics.CategoricalAccuracy()]
			}
		}
		for x in range(1, 3)]
	# Start the automated testing
	automatedTesting = t.AutomatedTesting(models, "Alzheimers-classification/data", "output", trainingArgs=trainingArgs)

	automatedTesting.start()

	# Check if the automated testing is done
	assert automatedTesting.done is True