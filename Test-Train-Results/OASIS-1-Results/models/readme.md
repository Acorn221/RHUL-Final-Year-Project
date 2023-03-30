## These training results were with the following config, on the Alzheimer's OASIS-1 Dataset:

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
            "other": {
                "fully_trainable": True,
            },
            "compile": {
                "optimizer": Adam(learning_rate=0.0000001),
                "loss": CategoricalCrossentropy(),
                "metrics": [CategoricalAccuracy()],
            },
        },
    ]