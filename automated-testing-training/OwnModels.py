import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as k
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

outDir = "./automated-testing-training/BasicModels/"

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

# Credits to https://stackoverflow.com/a/67138072/5758415
class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


# Making my own models from scratch as a point of comparison
def getModel(model):
  if(model == 1):
    return keras.Sequential([
      Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
      MaxPooling2D(2, 2),
      Conv2D(64, 3, activation='relu'),
      MaxPooling2D(2, 2),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(4, activation='sigmoid')
    ])
  elif(model == 2):
    return keras.Sequential([
      Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
      MaxPooling2D(2, 2),
      Conv2D(128, 3, activation='relu'),
      MaxPooling2D(2, 2),
      Conv2D(128, 3, activation='relu'),
      MaxPooling2D(2, 2),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(4, activation='sigmoid')
    ])
  elif(model == 3):
    return keras.Sequential([
      Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
      MaxPooling2D(2, 2),
      Conv2D(512, 3, activation='relu'),
      MaxPooling2D(2, 2),
      Conv2D(512, 3, activation='relu'),
      MaxPooling2D(2, 2),
      Conv2D(512, 3, activation='relu'),
      MaxPooling2D(2, 2),
      Flatten(),
      Dense(128, activation='relu'),
      Dense(4, activation='sigmoid')
    ])
  elif(model == 4):
    return keras.Sequential([
      Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
      Flatten(),
      Dense(512, activation='relu'),
      Dense(256, activation='relu'),
      Dense(512, activation='relu'),
      Dense(4, activation='sigmoid')
    ])
  elif(model == 5):
    return keras.Sequential([
      Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
      Flatten(),
      Dense(1024, activation='relu'),
      Dense(512, activation='relu'),
      Dense(1024, activation='relu'),
      Dense(4, activation='sigmoid')
    ])
  elif(model == 6):
    return keras.Sequential([
      Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
      Flatten(),
      Dense(1024, activation='relu'),
      Dense(512, activation='relu'),
      Dense(1024, activation='relu'),
      Dense(4, activation='sigmoid')
    ])


for i in range(4, 6):
  miscInfo = {}
  modelName = 'Model'+str(i)
  model = getModel(i)
  model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(0.001), metrics = ['accuracy'], run_eagerly=True)
  for j in range(0, 8):
    trainingStats = model.fit(train_ds,
      validation_data=test_ds,
      epochs=6, callbacks=ClearMemory())

    history = merge_dicts(history, trainingStats.history)	

  model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(0.0001), metrics = ['accuracy'], run_eagerly=True)
  
  for j in range(20):
    trainingStats = model.fit(train_ds,
      validation_data=test_ds,
      epochs=6, callbacks=ClearMemory())
    
    history = merge_dicts(history, trainingStats.history)

  miscInfo['BaseModelName'] = modelName
  saveInfo = {**history, **miscInfo}

  with open("./automated-testing-training/BasicModels/training-results/"+modelName+".json", "w+") as outfile:
    json.dump(saveInfo, outfile)

  model.save('./automated-testing-training/BasicModels/completed-models/'+modelName)

