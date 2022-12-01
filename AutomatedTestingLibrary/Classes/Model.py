import json
import time

# The directory where the history will be saved along with the models
modelDir = 'TrainedModels/'

# TODO: clear memory after each model is trained
# TODO: make sure everything is saved to the correct directory
# TODO: allow quick loading of models for quick testing/training
# TODO: allow quick and easy statistics to be generated
# TODO: allow for models to be easily trained after a crash or interruption

class Model:
	def __init__(self, modelCallback, modelname):
		self.modelCallback = modelCallback
		self.sequentialModel = None
		self.modelName = modelname
		self.history = None
		self.trained = False
		self.epochs = 0
		self.trainingTime = 0

	def loadModel(self):
		self.sequentialModel = self.modelCallback()

	def saveHistory(self):
		infoToSave = { 'history': self.history.history, 'epochs': self.epochs, 'trainingTime': self.trainingTime }
		with open(modelDir+self.modelName+".json", "w+") as fp:
				json.dump(infoToSave, fp)

		self.sequentialModel.save(modelDir+self.modelName+".h5")

	def mergeHistory(self, history):
		# Get the union of dict1 and dict2
		keys = set(self.history).union(history)
		arr = []
		self.history = dict((k, self.history.get(k, arr) + history.get(k, arr)) for k in keys)

	def compileModel(self, optimizer, loss, metrics):
		self.sequentialModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

	"""
		This middleware function is used to train the model
		It also saves the history of the model, 
	"""
	def fit(self, train_ds, test_ds, **kwargs):
		startTime = time.time()
		history = self.sequentialModel.fit(train_ds, validation_data=test_ds, **kwargs)
		self.mergeHistory(history.history)
		if hasattr(kwargs, 'epochs'):
			self.epochs += kwargs['epochs']

		self.trainingTime += time.time() - startTime

	def getSequentialModel(self):
		return self.sequentialModel

	def getHistory(self):
		return self.history
