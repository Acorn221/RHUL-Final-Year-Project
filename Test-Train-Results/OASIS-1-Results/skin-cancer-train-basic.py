import os
import numpy as np
import pandas as pd
from functools import partial
import AutomatedTestingLibrary.Classes.AutomatedTesting as t
import AutomatedTestingLibrary.Classes.Model as m
import tensorflow as tf
import tensorflow.keras as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Dropout
from keras import Sequential
from keras.applications import (
    MobileNetV2,
    InceptionV3,
    MobileNetV3Large,
    MobileNetV3Small,
    ResNet50V2,
    ResNet101V2,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB3,
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    VGG16,
    VGG19,
    Xception,
)
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy


"""
This script is used to generate the transfer learning results from the OASIS-1 dataset.
It does not use any other inputs, aside from the MRI scans.

"""

# Set directory paths
data_dir = 'C:\Active-Projects\RHUL-FYP\PROJECT\skin-cancer-dataset\Resized_200x200_MIX_2Classes'
model_dir = 'C:\Active-Projects\RHUL-FYP\PROJECT\Test-Train-Results\OASIS-1-Results\skin-cancer-400-epoch\\'

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 400
TF_SEED = 42

tf.random.set_seed(TF_SEED)

def create_model(base_model):
    model = Sequential()

    pretrained_model = base_model(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )

    for layer in pretrained_model.layers:
        layer.trainable = False

    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    return model


class AutomatedRegularTesting(t.AutomatedTesting):
    def loadTrainingData(self):
        data_generator = ImageDataGenerator(
          **self.augmentationParams,
        )

        self.train_ds = data_generator.flow_from_directory(
          data_dir,
          target_size=IMG_SIZE,
          batch_size=BATCH_SIZE,
          class_mode="categorical",
          shuffle=True,
          seed=42,
          subset="training",
        )

        self.test_ds = data_generator.flow_from_directory(
          data_dir,
          target_size=IMG_SIZE,
          batch_size=BATCH_SIZE,
          class_mode="categorical",
          shuffle=True,
          seed=42,
          subset="validation",
        )


if __name__ == "__main__":
    # Define the models to be tested
    models = []
    modelsToTest = [
        MobileNetV2,
        InceptionV3,
        MobileNetV3Large,
        MobileNetV3Small,
        ResNet50V2,
        ResNet101V2,
        EfficientNetB0,
        EfficientNetB1,
        EfficientNetB3,
        EfficientNetV2B0,
        EfficientNetV2B1,
        EfficientNetV2B2,
        VGG16,
        VGG19,
        Xception,
    ]

    for model in modelsToTest:
        callback = partial(create_model, model)
        models.append(m.Model(callback, modelname=model.__name__, saveDir=model_dir))

    # Define the augmentation parameters
    augmentationParams = dict(
        rescale=1.0 / 255,  # Scale the input in range (0, 1)
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        validation_split=0.2,
        rotation_range=360,
        brightness_range=(0.5, 1.5),
    )

    trainingArgs = [
        {
            "train": {
                "epochs": EPOCHS,
            },
            "compile": {
                "optimizer": Adam(learning_rate=0.000001),
                "loss": CategoricalCrossentropy(),
                "metrics": [CategoricalAccuracy()],
            },
        },
        {
            "train": {
                "epochs": EPOCHS,
            },
            "other": {
                "fully_trainable": True,
            },
            "compile": {
                "optimizer": Adam(learning_rate=0.00000001),
                "loss": CategoricalCrossentropy(),
                "metrics": [CategoricalAccuracy()],
            },
        },
    ]
    # Create the Automated Testing object
    testing = AutomatedRegularTesting(
        models, None, model_dir, augmentationParams, trainingArgs 
    )

    # Run the testing
    testing.start()
