# This is my project diary

## 19/10/2022

Today I have been deciding the specific classes of images I want to use for my project.
I had initially intended on using only 1 class, however I have decided to use 2 classes instead as I think it will be more interesting to see the results of the pruning on 2 classes rather than 1,
It will also demonstrate better the different strengths and weaknesses of the individual pruning methods.

I have decided to use the following sets of images:

- [Flowers Dataset](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)
	This dataset is likely to be an easier dataset for the models to get high accuracy on, as the different flowers are quite distinct from each other.

- [Alzheimer's Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)
	This dataset is going to be more difficult for the models as the changes are more subtle between the different scan images, and the images are not as distinct from each other.


I am going to follow the tutorial [here](https://www.tensorflow.org/tutorials/images/classification) which should help me to get a good understanding of how to use the datasets and how to train the models to start with.
I have no prior experience training models so this will be a good starting point for me.
I will then need to use parts of the code from the tutorial to transfer run the transfer learning.

## 26/10/2022

Today I have been working on the transfer learning tutorial, I've made more progress through it and I've managed to get
Mobilenetv3 to run on the flowers dataset, however I'm having some issues with the alzheimer's dataset.
This is significant progress as this was a major blocker for me as I was not sure how to go forward with the prediction.