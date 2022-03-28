import cv2
import numpy as np
import random, math

# resolution 640x360
height = 360
width = 640
gradient = np.zeros((height, width), dtype=np.uint8)

def calSpeed(xind, yind, speedX, speedY):
    if xind+squareW >= width-speedFactor or xind-speedFactor < 0:
        speedX *= -1
    if yind+squareW >= height-speedFactor or yind-speedFactor < 0:
        speedY *= -1
    return (xind + speedX, yind + speedY, speedX, speedY)


for x in range(width):
    gradient[:, x] = int((x/width) * 255)

speedFactor = 10
direction = random.random() * 2 * math.pi
(speedX, speedY) = (math.cos(direction) * speedFactor, math.sin(direction) * speedFactor)

squareW = 50
square = gradient[height//2:height//2+squareW, width//2:width//2+squareW]
yind = height//2-squareW//2
xind = width//2-squareW//2

def rect(pos, size, cutout):
    gradient[pos[0]:pos[0] + size[0], pos[1]:pos[1] + size[1]] = cutout

rect((50, 50), (squareW, squareW), square)
rect((50, 540), (squareW, squareW), square)

cv2.namedWindow('An interesting title', cv2.WINDOW_AUTOSIZE)

while True:
    result = gradient.copy()
    (xind, yind, speedX, speedY) = calSpeed(xind, yind, speedX, speedY)
    result[int(yind):int(yind+squareW), int(xind):int(xind+squareW)] = square
    
    cv2.imshow('An interesting title', result)

    if cv2.waitKey(10) == ord('q'):
        cv2.destroyAllWindows()
        break