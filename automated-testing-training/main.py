import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2, MobileNetV3Small, MobileNetV3Large, InceptionResNetV2, InceptionV3, ResNet101, VGG19, Xception
from sklearn.model_selection import GridSearchCV
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from pathlib import Path
import os.path
import json
import atexit

data_dir = Path('Alzheimers-classification/data/OriginalDataset');

train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.2,
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=360,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
) # set validation split

train_ds = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,

    class_mode='categorical',
    subset='training')

test_ds = train_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')

params = {
	"input_shape": (224,224,3),
	"include_top": False,
	"weights": 'imagenet'
}

outDir = "./automated-testing-training/training-results/"

modelName = ''
model = keras.Sequential()
history = {}


def save_history():
	global history
	global modelName
	global model
	with open(outDir+modelName+".json", "w+") as fp:
		json.dump(history, fp)

	model.save(outDir+modelName+".h5")


atexit.register(save_history)

def merge_dicts(dict1, dict2):
	# Get the union of dict1 and dict2
  keys = set(dict1).union(dict2)
  arr = []
  return dict((k, dict1.get(k, arr) + dict2.get(k, arr)) for k in keys)

pretrainedUnfrozenLayers = 5

models = [MobileNetV2, MobileNetV3Small, MobileNetV3Large, InceptionResNetV2, InceptionV3, ResNet101, VGG19]
modelNames = ["MobileNetV2", "MobileNetV3Small", "MobileNetV3Large", "InceptionResNetV2", "InceptionV3", "ResNet101", "VGG19"]

MobileNetV3Small = keras.applications.MobileNetV3Small(**params)

for i in range(len(models)):
	# Create the model

	miscInfo = {}
	pretrainedModel = models[i]
	modelName = modelNames[i]

	model = keras.Sequential()

	for layer in MobileNetV3Small.layers[:-pretrainedUnfrozenLayers]:
		layer.trainable = False

	model.add(pretrainedModel(**params))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Dense(512, activation='relu'))
	model.add(keras.layers.Dropout(0.5))
	model.add(keras.layers.Dense(4, activation='softmax'))
	model.summary()

	# Compile the model

	model.compile(
		optimizer=keras.optimizers.Adam(learning_rate=0.0001),
		metrics=['accuracy'],
		loss='binary_crossentropy',
	)

	history = {}
	# Train the model
	for j in range(0, 8):
		trainingStats = model.fit(
			train_ds,
			epochs=6, 
			steps_per_epoch=len(train_ds),
			validation_data=test_ds,
		)
		history = merge_dicts(history, trainingStats.history)

		if(j > 5):
			lastAccuracy = sum(history['accuracy'][:5])
			lastValAccuracy = sum(history['val_accuracy'][:5])

			if(lastAccuracy > (lastValAccuracy - 0.2)):
				print("Likely Overfitting, moving to fine tuning now")
				miscInfo['Overfitting excape'] = True
				break;

		if(history['val_accuracy'][:1][0] > 0.95):
			print("Val_accuracy is over 0.96, moving over to fine tuning")
			break;

	# Unfreeze all the layers
	for layer in model.layers:
		layer.trainable = True

	# Lower the training rate
	model.compile(optimizer=keras.optimizers.Adam(0.000001),
								loss=keras.losses.BinaryCrossentropy(),
								metrics=['accuracy'])

	# Train the model
	for j in range(20):
		trainingStats = model.fit(
			train_ds,
			epochs=6,
			steps_per_epoch=len(train_ds),
			validation_data=test_ds,
		)

		# Add the training statistics to the array
		history = merge_dicts(history, trainingStats.history)


	# Get the model name to save the data to

	# Set the misc info for reference
	miscInfo['params'] = params
	miscInfo['pretrainedUnfrozenLayers'] = pretrainedUnfrozenLayers
	miscInfo['BaseModelName'] = modelNames[i]
	saveInfo = {**history, **miscInfo}
 	# Save the data to a JSON file
	with open("./automated-testing-training/training-results/"+modelNames[i]+".json", "w+") as outfile:
		json.dump(saveInfo, outfile)

	# Save the model for future reference
	model.save('./automated-testing-training/completed-models/'+modelName)

