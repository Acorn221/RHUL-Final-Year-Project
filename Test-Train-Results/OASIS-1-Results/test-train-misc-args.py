import os
import numpy as np
import pandas as pd
from functools import partial
import AutomatedTestingLibrary.Classes.AutomatedTesting as t
import AutomatedTestingLibrary.Classes.Model as m
import tensorflow as tf
import tf.keras as keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Concatenate
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.layers import Flatten, Dense, Dropout
from keras import Sequential
from keras.applications import MobileNetV2, InceptionV3, MobileNetV3Large, MobileNetV3Small, ResNet50V2, ResNet101V2, EfficientNetB0, EfficientNetB1, EfficientNetB3, VGG16, VGG19, Xception
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.models import Model
from utils import createFileName, convertToString, mmseConvert

"""
This script is used to generate the transfer learning results from the OASIS-1 dataset, with the addition of the misc data.
It will use the extra data from the csv file, listed below:
 - age
 - gender, 
 - MMSE score (	Mini-Mental State Examination score), 
 - SES (Socioeconomic status)
 - eTIV (Estimated Total Intracranial Volume)
 - nWBV (Normalized Whole Brain Volume)


This additional data will be given to the data genrator and then passed to the model, in the hope that it will improve the results.
"""

scansDir = 'C:/Active-Projects/RHUL-FYP/PROJECT/OASIS/1/processed_scans'
csvFile = 'C:/Active-Projects/RHUL-FYP/PROJECT/OASIS/1/oasis_cross-sectional.csv'
modelDir = 'C:/Active-Projects/RHUL-FYP/PROJECT/Test-Train-Results/OASIS-1-Results/extra-data-models/'

"""
This function is used to create the model for the transfer learning, it takes in the base model and creates the surrounding layers
This is a function so that it can be used for the different models and avoid repeated code
"""
def create_model(base_model):

  # Creating the sub-model for handling the assocated data with the scan
  sub_model_input = Input(shape=(4,))
  x = Dense(64, activation='relu')(sub_model_input)
  x = Dense(512, activation='relu')(x)
  x = Dropout(0.2)(x)

  sub_model_output = Dense(64, activation='relu')(x)

  sub_model = Model(inputs=sub_model_input, outputs=sub_model_output)

  # Creating the model for handling the image data
  pretrained_model = base_model(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

  # Freezing the pretrained model
  for layer in pretrained_model.layers:
    layer.trainable = False

  inputs = Input(shape=(224, 224, 3))
  x = pretrained_model(inputs)
  x = Conv2D(64, (3, 3), activation='relu')(x)
  x = Flatten()(x)
  x = Dropout(0.5)(x)
  x = Concatenate()([sub_model.output, x])
  x = Dense(512, activation='relu')(x)
  outputs = Dense(4, activation='softmax')(x)

  # Clearly defining the inputs and outputs of the model
  model = Model(inputs=[sub_model_input, inputs], outputs=outputs)

  return model


