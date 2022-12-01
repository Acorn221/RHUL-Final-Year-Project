import Classes.AutomatedTesting as t
import Classes.Model as m
from tensorflow.keras import layers, Sequential
from tensorflow.keras.applications import MobileNetV2

def MobileNetModel():
		model = Sequential()
		mobileNet = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

		for layer in mobileNet.layers:
			layer.trainable = False

		model.add(mobileNet)
		model.add(layers.GlobalAveragePooling2D())
		model.add(layers.Dense(2, activation='softmax'))
		
		return model

def main():
	print(m.Model)
	models = [m.Model(MobileNetModel, "MobileNetV2")]

	automatedTesting = t.AutomatedTesting(models, "Alzheimers-classification/data", "output")

# Check if it's the main file
if __name__ == "__main__":
	# Run the main function
	main()