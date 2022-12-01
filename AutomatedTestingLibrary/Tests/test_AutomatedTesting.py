import AutomatedTestingLibrary.Classes.AutomatedTesting as t
import AutomatedTestingLibrary.Classes.Model as m

from utils import MobileNetModel

models = [m.Model(MobileNetModel, "MobileNetV2")]

automatedTesting = t.AutomatedTesting(models, "Alzheimers-classification/data", "output")

def test_load_training_data():
	# Load the training data
	automatedTesting.loadTrainingData()
	# Check if the training data is loaded
	assert automatedTesting.train_ds is not None
	assert automatedTesting.test_ds is not None