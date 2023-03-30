import os
import numpy as np
import pandas as pd
from functools import partial
import AutomatedTestingLibrary.Classes.AutomatedTesting as t
import AutomatedTestingLibrary.Classes.Model as m
import tensorflow as tf
import tf.keras as keras
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
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
    VGG16,
    VGG19,
    Xception,
)
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from utils import createFileName, convertToString, mmseConvert


"""
This script is used to generate the transfer learning results from the OASIS-1 dataset.
It does not use any other inputs, aside from the MRI scans.

The directories need to be set to the locations of the scans and the csv file, then the output directory and model directory need to be set to where the results should be saved.

"""

scansDir = "C:/Active-Projects/RHUL-FYP/PROJECT/OASIS/1/processed_scans"
csvFile = "C:/Active-Projects/RHUL-FYP/PROJECT/OASIS/1/oasis_cross-sectional.csv"
outputDir = (
    "C:/Active-Projects/RHUL-FYP/PROJECT/Test-Train-Results/OASIS-1-Results/output/"
)
modelDir = (
    "C:/Active-Projects/RHUL-FYP/PROJECT/Test-Train-Results/OASIS-1-Results/models/"
)

"""
    This create_model function is used to create the model that will be used for the transfer learning.
    It takes in a base model, which is the model that will be used as the base.
    This is used so that I do not have to create a new function for each model.
"""
def create_model(base_model):
    model = Sequential()

    pretrained_model = base_model(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )

    for layer in pretrained_model.layers:
        layer.trainable = False

    model.add(pretrained_model)
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation="softmax"))

    return model


"""
The AutomatedTesting class was designed to be inherited from, so that the loadTrainingData function can be overwritten.
This allows for the data to be loaded in a different way, like I need to here
"""
class AutomatedRegularTesting(t.AutomatedTesting):
    def __init__(
        self, models, data_dir, output, augmentationParams, trainingArgs, batchSize=32
    ):
        self.batchSize = batchSize
        super().__init__(models, data_dir, output, augmentationParams, trainingArgs)

    def loadTrainingData(self):
        # Importing the scans from processed_scans folder,
        fileNames = []
        # Create an array of all the file names from the processed_scans folder
        for root, dirs, files in os.walk(scansDir):
            for file in files:
                fileNames.append(file)

        # Importing the labels from the csv file
        labels = pd.read_csv(csvFile, dtype=str)

        # Checking if the file name is in the labels (there were problems without this)
        checkedLabels = labels[
            labels["ID"].isin([fileName[:13] for fileName in fileNames])
        ]

        checkedLabels["fileNames"] = checkedLabels["ID"].apply(createFileName)
        checkedLabels["CDR"] = checkedLabels["CDR"].apply(convertToString)

        # Print checked labels
        print(checkedLabels["fileNames"][0])

        # Splitting the data into training and testing

        data_generator = ImageDataGenerator(**self.augmentationParams)

        args = dict(
            dataframe=checkedLabels,
            directory=scansDir,  # This is required to have the full path for some reason
            x_col="fileNames",
            y_col="CDR",
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",  # Binary = it outputs 1 number, not the probability of each class
            shuffle=True,
            seed=420,
        )

        self.train_ds = data_generator.flow_from_dataframe(**args, subset="training")

        self.test_ds = data_generator.flow_from_dataframe(**args, subset="validation")


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
        VGG16,
        VGG19,
        Xception,
    ]

    for model in modelsToTest:
        # partial is used to pass in the model to the create_model function, but without calling it, otherwise the ram will be used up
        callback = partial(create_model, model)
        models.append(m.Model(callback, modelname=model.__name__, saveDir=modelDir))

    # Define the data directory
    data_dir = None

    # Define the augmentation parameters
    augmentationParams = {
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "shear_range": 0.2,
        "zoom_range": 0.2,
        "horizontal_flip": True,
        "fill_mode": "nearest",
        "validation_split": 0.2,
    }

    trainingArgs = [
        {
            "train": {
                "epochs": 100,
            },
            "compile": {
                "optimizer": Adam(learning_rate=0.00001),
                "loss": CategoricalCrossentropy(),
                "metrics": [CategoricalAccuracy()],
            },
        },
        {
            "train": {
                "epochs": 100,
            },
            "other": { # Fully trainable is used to make the model fully trainable, so that there are no frozen layers
                "fully_trainable": True,
            },
            "compile": {
                "optimizer": Adam(learning_rate=0.0000001),
                "loss": CategoricalCrossentropy(),
                "metrics": [CategoricalAccuracy()],
            },
        },
    ]
    # Create the Automated Testing object
    testing = AutomatedRegularTesting(
        models, data_dir, outputDir, augmentationParams, trainingArgs
    )

    # Run the testing
    testing.start()
