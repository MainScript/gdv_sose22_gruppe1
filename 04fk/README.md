# Documentation

## README
Python Version 3.10.2
### Packages
- numpy=1.22.3
- cv2=4.5.5
- math=3.10.2

## How to run the program
* Open repository in VS-Code
* Create a new folder called "data" as subfolder of the folder "04"
* Download the Caltech-101 image data set from https://data.caltech.edu/records/20086. It contains a folder called "101_ObjectCategories", extract the folders in it into the "data" folder.
* Run 04/main.py


## Program Sequence
* Creates new training data and saves it, or loads previouly created trainings data
* Loads input image
* Splits input image into tiles
* Uses the training data to find the best matching image from the data set for each tile
* Creates an image from the matches found
* Saves the resulting image
* Shows the input image and the resulting image until a key is pressed, then closes the program

## Example Images
Example Images are in the `input` and `output` folder.