## These training results were with the following config:

	def create_model(base_model):
			model = Sequential()

			pretrained_model = base_model(
					input_shape=(224, 224, 3), include_top=False, weights="imagenet"
			)

			for layer in pretrained_model.layers[:-5]:
					layer.trainable = False

			model.add(pretrained_model)
			model.add(Flatten())
			model.add(Dense(1024, activation="relu"))
			model.add(Dense(NUM_CLASSES, activation="softmax"))

			return model

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

			augmentationParams = dict(
        rescale=1.0 / 255,  # Scale the input in range (0, 1)
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        validation_split=0.2,
        rotation_range=360,
        brightness_range=(0.7, 1.3),
        
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