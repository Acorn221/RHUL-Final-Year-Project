import json
import time

# The directory where the history will be saved along with the models
modelDir = 'TrainedModels/'

class Model:
	def __init__(self, modelCallback, modelname):
			self.modelCallback = modelCallback
			self.sequentialModel = None
			self.modelName = modelname
			self.history = None
			self.trained = False
			self.epochs = 0
			self.trainingTime = 0

	def loadModel(self, sequentialModel):
			self.sequentialModel = self.modelCallback()

	def saveHistory(self):
			with open(modelDir+self.modelName+".json", "w+") as fp:
					json.dump(self.history, fp)

			self.sequentialModel.save(modelDir+self.modelName+".h5")

	def mergeHistory(self, history):
		# Get the union of dict1 and dict2
		keys = set(self.history).union(history)
		arr = []
		self.history = dict((k, self.history.get(k, arr) + history.get(k, arr)) for k in keys)

	def train(self, train_ds, test_ds, epochs):
		# Record the time it takes to train the model
		startTime = time.time()
		history = self.sequentialModel.fit(train_ds, epochs=epochs, validation_data=test_ds)
		self.mergeHistory(history.history)
		self.epochs += epochs
		self.trainingTime += time.time() - startTime

	def getSequentialModel(self):
		return self.sequentialModel

	def getHistory(self):
		return self.history
