import os
import numpy as np
import pandas as pd
from functools import partial
import AutomatedTestingLibrary.Classes.AutomatedTesting as t
import AutomatedTestingLibrary.Classes.Model as m
import tensorflow as tf
import tf.keras as keras
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input,
    Concatenate,
)
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
from keras.models import Model
from keras.utils import Sequence
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

scansDir = "C:/Active-Projects/RHUL-FYP/PROJECT/OASIS/1/processed_scans"
csvFile = "C:/Active-Projects/RHUL-FYP/PROJECT/OASIS/1/oasis_cross-sectional.csv"
modelDir = "C:/Active-Projects/RHUL-FYP/PROJECT/Test-Train-Results/OASIS-1-Results/extra-data-models/"

"""
This function is used to create the model for the transfer learning, it takes in the base model and creates the surrounding layers
This is a function so that it can be used for the different models and avoid repeated code
"""


def create_model(base_model):
    # Creating the sub-model for handling the assocated data with the scan
    sub_model_input = Input(shape=(4,))
    x = Dense(64, activation="relu")(sub_model_input)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)

    sub_model_output = Dense(64, activation="relu")(x)

    sub_model = Model(inputs=sub_model_input, outputs=sub_model_output)

    # Creating the model for handling the image data
    pretrained_model = base_model(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )

    # Freezing the pretrained model
    for layer in pretrained_model.layers:
        layer.trainable = False

    inputs = Input(shape=(224, 224, 3))
    x = pretrained_model(inputs)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Concatenate()([sub_model.output, x])
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(4, activation="softmax")(x)

    # Clearly defining the inputs and outputs of the model
    model = Model(inputs=[sub_model_input, inputs], outputs=outputs)

    return model


class MRIDataGenerator(Sequence):
    def __init__(
        self,
        df,
        datagen,
        batch_size,
        df_input_labels,
        df_target_label,
        target_size=(224, 224, 3),
        shuffle=True,
    ):
        self.df = df
        self.datagen = datagen
        self.df_input_labels = df_input_labels
        self.df_target_label = df_target_label
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        # Shuffle the data if shuffle is True
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx):
        # If the batch size is not a multiple of the total number of images, the last batch will be smaller
        batchEnd = (
            (idx + 1) * self.batch_size
            if (idx + 1) * self.batch_size < len(self.df)
            else len(self.df)
        )

        # Get the batch
        batch = self.df[idx * self.batch_size : batchEnd]

        # Create the arrays to store the images and labels
        batch_x = np.zeros((len(batch),) + self.target_size + (3,), dtype="float32")
        batch_y = np.zeros((len(batch),), dtype="int32")

        # Create the arrays to store the attributes
        batch_a = np.zeros((len(batch), len(self.df_input_labels)), dtype="float32")

        for i, row in batch.iterrows():
            try:
                # i refers to the index of the dataframe, not the batch
                current_index = i % self.batch_size - 1

                # print(f"i: {i}, currentIndex: {currentIndex}, batchEnd: {batchEnd}, batchStart: {idx*self.batch_size}")

                img = tf.keras.preprocessing.image.load_img(
                    row["fileNames"], target_size=self.target_size
                )

                # Convert the image to an array
                x = tf.keras.preprocessing.image.img_to_array(img)

                # Apply the data augmentation
                x = self.datagen.random_transform(x)
                x = self.datagen.standardize(x)

                # Add the image to the batch
                batch_x[current_index] = x

                # Add the label to the batch
                batch_y[current_index] = float(row[self.df_target_label])

                # Add the attributes to the batch
                for label, i in self.df_input_labels:
                    batch_a[current_index, i] = float(row[label])

            except Exception as e:
                print(f"Error with Datagen For Loop: {e}")

        return [batch_a, batch_x], batch_y
