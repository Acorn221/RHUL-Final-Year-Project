from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

class AutomatedTesting:
	"""
		This will be the main class that will be used to run the automated testing
		It will take a list of models 
	"""
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
						"epochs": 10,
						"steps_per_epoch": 1,
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
				self.currentModel.compileModel(**arg["compile"])
				self.currentModel.fit(self.train_ds, self.test_ds, **arg["train"])
			self.currentModel.saveModel(self.output)
			self.currentModel.saveHistory()
			self.currentModel = None

		self.done = True


