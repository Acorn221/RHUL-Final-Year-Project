
# Import the model
from tensorflow.keras import layers, Sequential

# Import the MobileNetV2 model

from tensorflow.keras.applications import MobileNetV2

# Create a function that will return the model
def MobileNetModel():
	# Create a sequential model
	model = Sequential()
	# Create a MobileNetV2 model
	mobileNet = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
	# Set the layers to not be trainable
	for layer in mobileNet.layers:
		layer.trainable = False
	# Add the MobileNetV2 model to the sequential model
	model.add(mobileNet)
	# Add a global average pooling layer
	model.add(layers.GlobalAveragePooling2D())
	# Add a dense layer with 2 nodes and a softmax activation function
	model.add(layers.Dense(2, activation='softmax'))
	# Return the model
	return model
