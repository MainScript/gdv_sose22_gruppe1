# Documentation

## README
Python Version 3.10.2
### Packages
- numpy=1.22.3
- cv2=4.5.5
- math=3.10.2

## How to run the program
* Open repository in VS-Code
* Download the Caltech-101 image data set from https://data.caltech.edu/records/20086. It contains a folder called "101_ObjectCategories", extract that folder into the "data" folder.
* When running the programm for the first time (or wanting to create new training data), createNewTrainingData in main.py has to be set to True. Can be set to False next time running the progam, it then loads and uses the previously created training data.
* When setting saveResult in main.py to true, the resulting image gets saved to the file specified in the saveResultTo variable


## Program Sequence
* Creates new training data and saves it, or loads previouly created trainings data
* Loads input image
* Splits input image into tiles
* Uses the training data to find the best matching image from the data set for each tile
* Creates an image from the matches found
* Saves the resulting image (if saveResult is set to True)
* Shows the input image and the resulting image until a key is pressed, then closes the program


## Example Images
### Example 1
Coming soon
### Example 2
Coming soon
