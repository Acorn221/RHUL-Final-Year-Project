import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2, MobileNetV3Small, MobileNetV3Large, InceptionResNetV2, InceptionV3, ResNet101, VGG19
from sklearn.model_selection import GridSearchCV
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from pathlib import Path
import os.path

data_dir = Path('Alzheimers-classification/data/AugmentedAlzheimerDataset/');

train_datagen = ImageDataGenerator(rescale=1./255,
    validation_split=0.2) # set validation split

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

pretrainedUnfrozenLayers = 5

models = [MobileNetV2, MobileNetV3Small, MobileNetV3Large, InceptionResNetV2, InceptionV3, ResNet101, VGG19]

MobileNetV3Small = keras.applications.MobileNetV3Small(**params)

for pretrainedModel in models:
	# Create the model



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

	# Train the model
	history = model.fit(
		train_ds,
		epochs=2, 
		steps_per_epoch=len(train_ds),
		validation_data=test_ds,
	)

	print("History:", history.history['accuracy'])
	print(history.history['val_accuracy'])
	print("History:", history.history)
"""
# Make all of the model layers trainable
for layer in model.layers:
	layer.trainable = True

model.compile(optimizer=keras.optimizers.Adam(0.000001),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(
  train_ds,
  epochs=20,
  steps_per_epoch=len(train_ds),
  validation_data=test_ds,
)"""