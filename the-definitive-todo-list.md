# Things I need to do

## At home

- [x] Setup the OASIS dataset training
  - [x] Clear some space on my laptop 
  - [x] Download and extract the dataset
  - [x] Find out what parts in the dataset I can use
    - [x] Avoid bias by giving it the age of the patient
    - [x] Check for duplicates
- [ ] Fix the training with the OASIS-1 dataset
  - [ ] Find out why it always predicts 1
  - [ ] See if classification or regression is better for it
- [ ] Setup the OASIS-3 Dataset
  - [ ] Download and extract the dataset
  - [ ] Find out how to read and interpret the dataset
  - [ ] Parse the dataset into a usable format, to be used by keras
  - [ ] See if training with the OASIS-3 dataset is better
- [ ] Use better methods of analysis for the performance
  - [ ] Confusion Matrices
- [ ] Hyperparameter tuning
	- [ ] Use a grid search
	- [ ] Use a random search
	- [ ] Look more into hyperparameter tuning and model optimisation
- [ ] use googlenet or efficentnet too

## On campus

- [ ] Finish the automated testing library
  - [ ] Write tests for the model memory clearing
    - [ ] Write the model memory clearing function
  - [ ] Write tests for hyperparameter tuning
		- [ ] Write the hyperparameter tuning function
  - [ ] Allow for the image input and the other related parameters to be passed into the model
    - [ ] Test
    - [ ] Function
- [ ] Create a web interface for uploading MRI scans and getting the results
  - [ ] Set up a flask API
    - [ ] Create tests for the API
  - [ ] Create a basic web interface for uploading and getting the predictions
- [ ] The Report
  - [ ] The Diary
  - [ ] Document the different hyperparameter tuning
  - [ ] Document the different ways of testing the model and showing its performance
  - [ ] Talk about the web interface and it's potential uses
- [ ] Look for another dataset (medical imaging)
  - [x] OASIS-3 - Gained access to the dataset