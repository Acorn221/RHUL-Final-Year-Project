import Classes.AutomatedTesting as t
import Classes.Model as m
from tensorflow.keras import layers, Sequential, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2, InceptionModel

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
		inception = InceptionModel(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

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