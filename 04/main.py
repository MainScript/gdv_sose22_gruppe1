from turtle import width
import cv2
import numpy as np
import math
from GDV_TrainingSet import Descriptor, TrainingSet

# TODO include edges in creating image from tiles
# TODO save result with button press, number images automatically, import os
# TODO cleanup and optimize

# SETTINGS
img_path = "./04/images/image1.jpg"
tile_size = (32, 32)    # height, width
saveResult = True

# Creating new training data
createNewTrainingData = True    # creates and saves new training data, old gets overwritten
descriptor = Descriptor.TINY_COLOR32
trainDataFolder = './04/data/101_ObjectCategories/'
trainDataFile = './04/data/data.npz'

def GetTile(img, y, x, tile_width, tile_height):
    x_start = x * tile_width
    x_end = min(x_start + tile_width, img.shape[1])
    y_start = y * tile_height
    y_end = min(y_start + tile_height, img.shape[0])
    return img[y_start: y_end, x_start: x_end]


def CreateImageFromTiles(tiles, height, width, tile_size):
    output = np.zeros((height + 1, width + 1, 3), np.uint8)
    for y in range(tiles.shape[0] - 1):
        for x in range(tiles.shape[1] - 1):
            tile = cv2.resize(tiles[y, x], tile_size)
            x_start = x * tile_size[1]
            x_end = x_start + tile.shape[1]
            y_start = y * tile_size[0]
            y_end = y_start + tile.shape[0]
            output[y_start: y_end, x_start: x_end] = tile
    return output


def findBestMatch(trainData, sample):
    # do the matching with FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(trainData.trainData, sample, k=1)
    # Sort by their distance.
    matches = sorted(matches, key=lambda x: x[0].distance)
    bestMatch = matches[0][0]
    return bestMatch.queryIdx


def convertTile(tile):
    global descr, trainData
    newcomer = np.ndarray(shape=(1, descr.getSize()),
                      buffer=np.float32(descr.compute(tile)),
                      dtype=np.float32)
    idx = findBestMatch(trainData, newcomer)
    return cv2.imread(trainData.getFilenameFromIndex(idx), cv2.IMREAD_COLOR)


# load input image
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
h, w, d = img.shape

# save tiles to array
tile_rows = math.ceil(h / tile_size[0])
tile_cols = math.ceil(w / tile_size[1])

tiles = np.ndarray((tile_rows, tile_cols), np.ndarray)
for y in range(tile_rows):
    for x in range(tile_cols):
        tiles[y, x] = GetTile(img, y, x, tile_size[0], tile_size[1])


# handle training data
root_path = './04/data/101_ObjectCategories/'
file_name = './04/data/data.npz'
trainData = TrainingSet(root_path)

if createNewTrainingData:
    trainData.createTrainingData(Descriptor.TINY_COLOR32)
    trainData.saveTrainingData(file_name)
else:
    trainData.loadTrainingData(file_name)


assert(isinstance(trainData.descriptor, Descriptor))
descr = trainData.descriptor

# find matches for each tile and save them to array
tile_matches = np.ndarray(tiles.shape, np.ndarray)
for y in range(tiles.shape[0]):
    for x in range(tiles.shape[1]):
        tile_matches[y, x] = convertTile(tiles[y, x])


result = CreateImageFromTiles(tile_matches, img.shape[0], img.shape[1], tile_size)
if saveResult:
    cv2.imwrite("./04/images/result.png", result)
cv2.imshow("tile_match", result)
cv2.imshow("original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
