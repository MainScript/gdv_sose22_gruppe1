import cv2
import numpy as np
import math
from GDV_TrainingSet import Descriptor, TrainingSet

# SETTINGS
img_path = "./04/images/image1.jpg"                         # location of the input image
tile_size = (150, 150)                                      # (height, width) size of the tiles
saveResult = True                                           # saves the resulting image

createNewTrainingData = False                               # creates and saves new training data if true
descriptor = Descriptor.TINY_COLOR32                        # descriptor used for creating the new training data
trainingDataFolder = './04/data/101_ObjectCategories/'      # location of the data set used for creating the training data
trainingDataFile = './04/data/data.npz'                     # location of the training data file


# splits image into tiles according to tile_size
# then returns the tiles as array
def getTilesFromImage(img, tile_size):
    h, w, d = img.shape
    tile_rows = math.ceil(h / tile_size[0])
    tile_cols = math.ceil(w / tile_size[1])

    tiles = np.ndarray((tile_rows, tile_cols), np.ndarray)
    for y in range(tile_rows):
        for x in range(tile_cols):
            x_start = x * tile_size[1]
            x_end = min(x_start + tile_size[1], img.shape[1])
            y_start = y * tile_size[0]
            y_end = min(y_start + tile_size[0], img.shape[0])
            tiles[y, x] = img[y_start: y_end, x_start: x_end]
    return tiles


def createImageFromTiles(tiles, tile_size, result_size):
    result = np.zeros((result_size[0], result_size[1], 3), np.uint8)
    for y in range(tiles.shape[0]):
        for x in range(tiles.shape[1]):
            tile = cv2.resize(tiles[y, x], tile_size)
            x_start = x * tile_size[1]
            x_end = x_start + tile.shape[1]
            y_start = y * tile_size[0]
            y_end = y_start + tile.shape[0]
            result[y_start: y_end, x_start: x_end] = tile
    return result
    

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


# finds the best match for each tile and saves them to an array
def getBestTileMatches(tiles, trainData):
    descriptor = trainData.descriptor
    best_matches = np.ndarray(tiles.shape, np.ndarray)
    for y in range(tiles.shape[0]):
        for x in range(tiles.shape[1]):
            tile = np.ndarray(shape=(1, descriptor.getSize()),
                      buffer=np.float32(descriptor.compute(tiles[y, x])),
                      dtype=np.float32)
            idx = findBestMatch(trainData, tile)
            best_match = cv2.imread(trainData.getFilenameFromIndex(idx), cv2.IMREAD_COLOR)
            best_matches[y, x] = best_match
    return best_matches


# handle training data
trainData = TrainingSet(trainingDataFolder)
if createNewTrainingData:
    trainData.createTrainingData(descriptor)
    trainData.saveTrainingData(trainingDataFile)
else:
    trainData.loadTrainingData(trainingDataFile)

# load input image
input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
# split image into tiles
tiles = getTilesFromImage(input_img, tile_size)
# find the best matches for each tile
result_tiles = getBestTileMatches(tiles, trainData)
# recreate the image from the found matching tiles
result = createImageFromTiles(result_tiles, tile_size, input_img.shape)

# save image
if saveResult:
    cv2.imwrite("./04/images/result.png", result)

# show input and resulting image
cv2.imshow("tile_match", result)
cv2.imshow("original", input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
