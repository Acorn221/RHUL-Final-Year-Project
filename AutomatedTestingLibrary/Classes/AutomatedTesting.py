from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import gc
from keras import backend as k
from keras.callbacks import Callback

"""
	This is the main class for the automated testing
	It loads 1 model at a time, trains it, then saves it
	This will save me time when I want to test a new model or set of models
	It allows hyperparameter tuning with the training args
"""
class AutomatedTesting:
	def __init__(self, models, data_dir, output, augmentationParams=None, trainingArgs=None):
		self.currentModel = None
		self.models = models
		self.data_dir = data_dir
		self.output = output
		self.train_ds = None
		self.test_ds = None
		self.done = False
		
		# Setting the training arguments to the default if none are provided
		if trainingArgs is None:
			self.trainingArgs = [
						{
					"train": {
						"epochs": 5,
						"steps_per_epoch": 5,
					},
					"compile": {
						"optimizer": keras.optimizers.Adam(learning_rate=0.001),
						"loss": keras.losses.CategoricalCrossentropy(),
						"metrics": [keras.metrics.CategoricalAccuracy()]
					}
				}
			]
		else:
			self.trainingArgs = trainingArgs

		# Setting the augmentation parameters to the default if none are provided
		if augmentationParams is not None:
			self.augmentationParams = augmentationParams
		else:
			self.augmentationParams = {
				"validation_split": 0.2,
				"featurewise_center": True,
				"featurewise_std_normalization": True,
				"rotation_range": 360,
				"width_shift_range": 0.1,
				"shear_range": 0.1,
				"zoom_range": 0.1,
				"height_shift_range": 0.1,
				"horizontal_flip": True,
				"vertical_flip": True,
			}
		self.loadTrainingData()

	"""
		This loads the training data from the given directory,
		after rescaling the images and augmenting them
	"""
	def loadTrainingData(self):
		train_datagen = ImageDataGenerator(rescale=1./255,
			**self.augmentationParams)
			
		self.train_ds = train_datagen.flow_from_directory(
			self.data_dir,
			target_size=(224, 224),
			batch_size=32,
			class_mode='categorical',
			subset='training')
		self.test_ds = train_datagen.flow_from_directory(
			self.data_dir,
			target_size=(224, 224),
			batch_size=32,
			class_mode='categorical',
			subset='validation')
			
	"""
		Called when a new model needs to be loaded, this will load the next model in the list
	"""
	def loadModel(self, model=None):
		if model is None:
			model = self.models.pop(0)
			model.loadModel()

		self.currentModel = model
	
	"""
		This starts the automated testing
		It will load the next model in the list and train it
		It will stop when there are no more models in the list
	"""
	def start(self):
		while len(self.models) > 0:
			self.loadModel()
			# Loop through all the training arguments, and train the model with each set of arguments
			for arg in self.trainingArgs:
				if 'other' in arg:
					if 'fullyTrainable' in arg["other"]:
						if arg["other"]["fullyTrainable"]:
							for layer in self.currentModel.sequentialModel.layers:
								layer.trainable = True

				self.currentModel.compileModel(**arg["compile"])
				self.currentModel.fit(self.train_ds, self.test_ds, **arg["train"])
			self.currentModel.saveHistory()
			gc.collect()
			k.clear_session()
			self.currentModel = None

		self.done = True

