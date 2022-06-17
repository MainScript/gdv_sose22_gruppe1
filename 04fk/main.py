from GDV_TrainingSet import Descriptor, TrainingSet
import os
import cv2
import numpy as np
import math

input_dir = './input/'
data_dir = './data/'
tilesize = (8, 8)
output_dir = './output/'


def load_data():
    trainingData = data_dir + 'data.npz'
    training_set = TrainingSet(data_dir)
    if not os.path.isfile(trainingData):
        print('Creating training data...')
        training_set.createTrainingData(Descriptor.TINY_COLOR32)
        print('Saving training data...')
        training_set.saveTrainingData(trainingData)
        print('Done.')
    else:
        print('Loading training data...')
        training_set.loadTrainingData(trainingData)
        print('Done.')

    return training_set


def make_tiles(img, tilesize):
    height, width, _ = img.shape
    tile_shape = (
        math.ceil(height / tilesize[0]), math.ceil(width / tilesize[1]))
    tiles = np.ndarray(tile_shape, np.ndarray)

    for rows in range(tile_shape[0]):
        for cols in range(tile_shape[1]):
            x_start = cols * tilesize[1]
            x_end = min(x_start + tilesize[1], width)
            y_start = rows * tilesize[0]
            y_end = min(y_start + tilesize[0], height)
            tiles[rows, cols] = img[y_start: y_end, x_start: x_end]
    print('Rows, Cols:', tile_shape)
    return tiles


def find_best_match_for_tile(tile, trainings_set):
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=200)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann_matcher.knnMatch(trainings_set.trainData, tile, k=1)
    matches = sorted(matches, key=lambda x: x[0].distance)
    best_match = cv2.imread(trainings_set.getFilenameFromIndex(
        matches[0][0].queryIdx), cv2.IMREAD_COLOR)
    return best_match


def tile_allocator(img, training_set):
    tiles = make_tiles(img, tilesize)
    descriptor = training_set.descriptor
    for row in range(tiles.shape[0]):
        print('Row:', row)
        for col in range(tiles.shape[1]):
            tile = np.ndarray(shape=(1, descriptor.getSize()),
                              buffer=np.float32(
                                  descriptor.compute(tiles[row, col])),
                              dtype=np.float32)
            best_match = find_best_match_for_tile(tile, training_set)
            tiles[row, col] = best_match
    return tiles


def create_image_from_tiles(tiles, result_size):
    result = np.zeros((result_size[0], result_size[1], 3), np.uint8)
    for y in range(tiles.shape[0]):
        for x in range(tiles.shape[1]):
            tile = cv2.resize(tiles[y, x], tilesize)
            x_start = x * tilesize[1]
            x_end = x_start + tile.shape[1]
            y_start = y * tilesize[0]
            y_end = y_start + tile.shape[0]
            result[y_start: y_end, x_start: x_end] = tile
    return result


training_set = load_data()
for file in os.listdir(input_dir):
    if file.endswith(".png") or file.endswith(".jpg"):
        input_img = cv2.imread(input_dir + file, cv2.IMREAD_COLOR)
        # use tile_allocator to create a tiles array and save it to output_dir
        # show the resulting image
        tiles = tile_allocator(input_img, training_set)
        output_img = create_image_from_tiles(tiles, input_img.shape)
        cv2.imshow('result', output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(output_dir + file, output_img)
        print('Saved ' + file)
