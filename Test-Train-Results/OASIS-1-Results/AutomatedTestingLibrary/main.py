import Classes.AutomatedTesting as t
import Classes.Model as m
import tensorflow
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.applications import MobileNetV2, InceptionV3

# This is an example file of how the library would be used

# The models have to be passed as a list of callable functions that return a model
# This is because otherwise if they were all loaded at the same time, they would take up all the memory

def MobileNetModel():
		model = Sequential()
		mobileNet = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

		for layer in mobileNet.layers:
			layer.trainable = False

		model.add(mobileNet)
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(4, activation='softmax'))
		
		return model

def InceptionModel():
		model = Sequential()
		inception = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

		for layer in inception.layers:
			layer.trainable = False

		model.add(inception)
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(4, activation='softmax'))
		
		return model



def main():
	print(m.Model)
	models = [m.Model(MobileNetModel, "MobileNetV2"), m.Model(InceptionModel, "InceptionV3")]

	automatedTesting = t.AutomatedTesting(models, "Alzheimers-classification/data", "output")
	automatedTesting.loadTrainingData()
	automatedTesting.start()

# Check if it's the main file
if __name__ == "__main__":
	# Run the main function
	main()