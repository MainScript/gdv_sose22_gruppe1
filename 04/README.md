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
![Example 1 input](/04/images/example1.jpg "Example 1 - Input (1200 x 1200)")
![Example 1 result 8](/04/images/example1result8.png "Example 1 - Result (Tile Size = 8))
![Example 1 result 16](/04/images/example1result16.png "Example 1 - Result (Tile Size = 16))
![Example 1 result 32](/04/images/example1result32.png "Example 1 - Result (Tile Size = 32))