import cv2
import numpy as np
import os


def opening(img, size, shape):
    kernel = cv2.getStructuringElement(shape, (size, size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)


def closing(img, size, shape):
    kernel = cv2.getStructuringElement(shape, (size, size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def getLowerRangeArray(hsv, range):
    return np.array([hsv[0] - range[0], hsv[1] - range[1], hsv[2] - range[2]])


def getUpperRangeArray(hsv, range):
    return np.array([hsv[0] + range[0], hsv[1] + range[1], hsv[2] + range[2]])


print("Use p to get the amount of chewing gums of the current color")
print("Use P to get the amount for each color of the current image")
print("Use o/O to cycle through the images")

# define hsv values and range
colors_hsv = {
    "blue": (109, 165, 235),
    "green": (85, 128, 161),
    "pink": (4, 127, 223),
    "red": (180, 210, 110),
    "white": (77, 0, 255),
    "yellow": (27, 217, 254)
}

hsv_ranges = {
    "blue": (50, 100, 100),
    "green": (50, 70, 30),
    "pink": (2, 83, 39),
    "red": (50, 100, 100),
    "white": (50, 70, 30),
    "yellow": (8, 69, 2)
}

colors = ("blue", "green", "pink", "red", "white", "yellow")

# kernel settings
morph_shape = cv2.MORPH_ELLIPSE
kernel_size = 5

# connectivity
connectivity = 8

# get all images
images = []

# gets all .jpg images from specified folder
path = "./02/images"

for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".jpg"):
            image = cv2.imread(os.path.join(root, file))
            images.append(image)


amount_colors = len(colors_hsv)
amount_images = len(images)
masks = np.ndarray((amount_images, amount_colors), np.ndarray)
amount = np.ndarray((amount_images, amount_colors), np.uint)
# generate mask, apply filters and count chewing gums
for i in range(amount_images):
    # create mask
    hsv = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)

    # generate masks and count for current image for every color
    for k in range(amount_colors):
        # generate mask for each colors
        lower = getLowerRangeArray(colors_hsv[colors[k]],
                                   hsv_ranges[colors[k]])
        upper = getUpperRangeArray(colors_hsv[colors[k]],
                                   hsv_ranges[colors[k]])
        mask = cv2.inRange(hsv, lower, upper)

        # apply filters to mask
        mask = opening(mask, kernel_size, morph_shape)
        mask = closing(mask, kernel_size, morph_shape)
        # save mask to array
        masks[i, k] = mask

        # count
        output = cv2.connectedComponentsWithStats(
                mask, connectivity, cv2.CV_32S)

        (numLabels, labels, stats, centroids) = output

        # check for size
        min_size = 10
        numRejected = 1
        for m in range(1, numLabels):
            w = stats[m, cv2.CC_STAT_WIDTH]
            h = stats[m, cv2.CC_STAT_HEIGHT]
            if w < min_size or h < min_size:
                numRejected += 1

        amount[i, k] = numLabels - numRejected
        mask = cv2.putText(mask, colors[k], (0, 20), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (255, 255, 255), 2)


current_image_index = 0
current_color_index = 0
while True:
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    # cycle through images
    # previous image
    if key == ord("o"):
        if current_image_index == 0:
            current_image_index = len(images) - 1
        else:
            current_image_index -= 1
    # next image
    if key == ord("O"):
        current_image_index = (current_image_index + 1) % len(images)

    # cycle through color masks
    # previous color
    if key == ord("c"):
        if current_color_index == 0:
            current_color_index = amount_colors - 1
        else:
            current_color_index -= 1
    # next color
    if key == ord("C"):
        current_color_index = (current_color_index + 1) % amount_colors

    # show amount of currently selected color
    if key == ord("p"):
        print(colors[current_color_index] + " " +
              str(amount[current_image_index, current_color_index]))
    # show amount for each color of current image
    if key == ord("P"):
        for i in range(amount_colors):
            print(colors[i] + " " + str(amount[current_image_index, i]))

    # show images
    cv2.imshow("mask", masks[current_image_index, current_color_index])
    cv2.imshow("original", images[current_image_index])


cv2.destroyAllWindows()
