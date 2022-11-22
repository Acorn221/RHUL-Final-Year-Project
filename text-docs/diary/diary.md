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

## 01/11/2022

Today I've worked more on the transfer learning tutorial and it can now identify the different flowers in the flowers dataset, 
with varying degrees of accuracy, around 80% was the best I got however this was only after 5 epochs and was the first time I have tried transfer learning.


## 09/11/2022

I had lost some of my git history for my diary, so I have had to rewrite some of the diary entries.
I have found more flower dataset images here and I've decided to change over to this as soon as possible;
https://www.kaggle.com/competitions/tpu-getting-started/overview

## 14/11/2022

I am training the model "Mobilenetv3 Small" with the alzheimer's dataset, I have been able to get it to run and it is currently training.
The power required to train the models is making it difficult to use trial and error for the parameters, so I am going to try and find a way to use the GPU on my laptop to train the models as soon as possible.
I'm also going to optimise the way I am training the models as to not use more power than necessary. This means I've got to get a better understanding of the models I am training with and how they work.
I have found this paper [here](https://arxiv.org/abs/1905.02244) which I think will be useful for me to read through and understand the structure of Mobilenetv3.

## 15/11/2022

I've had trouble getting the Alzheimers classification to work with transfer learning, so I decided to try and run other people's code to see if I could get it to work.
I have found that my transfer learning could be significantly more efficient and accurate if I use the existing code, so I'll have to try and identify where I went wrong with my code and make improvements.

## 21/11/2022

Today I managed to get the accuracy up to 90% on the alzheimer's dataset, this is a significant improvement from the 60% I was getting before. My model appears to still be overfitting however this progress is good. This was done through transfer learning on the Mobilenetv3 small model.

Output from the model during fine-tuning:
loss: 0.0314 - accuracy: 0.9883 - val_loss: 0.1337 - val_accuracy: 0.9036