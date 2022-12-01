from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

class AutomatedTesting:
	"""
		This will be the main class that will be used to run the automated testing
		It will take a list of models 
	"""
	def __init__(self, models, data_dir, output, augmentationParams=None):
		self.currentModel = None
		self.models = models
		self.data_dir = data_dir
		self.output = output
		self.train_ds = None
		self.done = False
		self.test_ds = None
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

		self.currentModel = model
	
	"""
		This starts the automated testing
		It will load the next model in the list and train it
		It will stop when there are no more models in the list
	"""
	def start(self):
		while len(self.models) > 0:
			self.loadModel()
			self.currentModel.loadModel()
			self.currentModel.compileModel(optimizer=keras.optimizers.Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
			self.currentModel.train(self.train_ds, self.test_ds, 10)
			self.currentModel.compileModel(optimizer=keras.optimizers.Adam(0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
			self.currentModel.train(self.train_ds, self.test_ds, 10)
			self.currentModel.saveHistory()
			self.currentModel = None

		self.done = True


