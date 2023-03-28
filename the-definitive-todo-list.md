# Things I need to do

## At home

- [x] Setup the OASIS dataset training
  - [x] Clear some space on my laptop 
  - [x] Download and extract the dataset
  - [x] Find out what parts in the dataset I can use
    - [x] Avoid bias by giving it the age of the patient
    - [x] Check for duplicates
- [ ] Fix the training with the OASIS-1 dataset
  - [x] Find out why it always predicts 1
    - The model just needs more information, it's not learning well with just the MRI scans and standard transfer learning
  - [x] See if classification or regression is better for it
    - Classification appears to yeild better results, as the dataset essentially has just 4 classes
- [z] Get the training with the MRI image and the other data in the OASIS-1 dataset
  - [x] Create a sub-model for the other data
  - [x] Combine the two models
  - [x] Feed the data to the new model, in the correct format
- [ ] User Interface
  - [ ] Make the interface work
  - [ ] Skin Cancer
    - [ ] Let the user take a photo with their phone
    - [ ] Process the image with the best performing model
    - [ ] Display the results
    - [ ] Host on github pages
  - [ ] Alzheimer's
    - [ ] Make the server side work with the model
- [ ] Try and predict the age of the patients to see if it's possible
  - [ ] See if the model can predict the age of the patients
  - [ ] Compare the impact of changing the input age to the model, to see how the model reacts
- [ ] Use better methods of analysis for the performance
  - [ ] Confusion Matrices
- [ ] Hyperparameter tuning
	- [ ] Use a grid search
	- [ ] Use a random search
	- [ ] Look more into hyperparameter tuning and model optimisation
- [ ] Setup the OASIS-3 Dataset
  - [ ] Download and extract the dataset
  - [x] Find out how to read and interpret the dataset
  - [ ] Parse the dataset into a usable format, to be used by keras
  - [ ] See if training with the OASIS-3 dataset is better


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
  - [x] Set up a flask API
    - [x] Create tests for the API
  - [x] Create a basic web interface for uploading and getting the predictions
  - [ ] Display the predictions
  - [ ] Actually get the predictions from the model
- [ ] The Report
  - [ ] The Diary
  - [ ] Document the different hyperparameter tuning
  - [ ] Document the different ways of testing the model and showing its performance
  - [ ] Talk about the web interface and it's potential uses
- [ ] Look for another dataset (medical imaging)
  - [x] OASIS-3 - Gained access to the dataset
- [ ] Talk about different metrics
  - [ ] Sensitivity/recall
  - [ ] Specificity
  - [ ] Precision
  - [ ] F1 Score



## Markscheme Checklist


- [ ] Contents and Knowledge (20%): Description of relevant theory - whether mathematical, algorithmic, hardware or software oriented. Also adequate chapters on development and Software Engineering;
- [ ] Critical analysis and Discussion (10%): A discussion of actual project achievements and how successful the project was. Clear evidence of reflection on the project process, its difficulties, successes and future enhancements. Any conclusions or results analysed or discussed appropriately;
  - Conclusion needs writing and the discussion needs to be expanded
  - The analysis of the results needs to be expanded and more need to be added to it
- [ ] Technical Decision Making (10%): Are important (technical) decisions well made and argued? This includes good design decisions, choice or development of algorithms, scope of the project.
  - Need to go over the technical decisions and make sure they are all justified
- [ ] Structure and Presentation (20%): Good use of English. Clear and appropriate report structure. Nice use of figures.
  - More graphs are needed
- [x] Bibliography and Citations (5%): Clear and appropriate bibliography with good citations. Must be clear and well formatted.
  - Bibtex has taken care of this
- [x] Professional issues (10%): Should be a topic relevant to the project undertaken.
  - [ ] I feel as though I have covered this well
- [x] Rationale (10%): Aims, objectives and a good introduction describing the structure of the report.
  - Added a rationale section, needs more objectives
- [x] Literature Review and Background Reading (15%): Description and critical analysis of relevant background material from books, research papers or the web. Analysis of existing systems that solve similar tasks;