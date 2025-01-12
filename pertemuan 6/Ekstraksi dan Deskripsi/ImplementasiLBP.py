import cv2
import numpy as np
from skimage.feature import local_binary_pattern

image = cv2.imread('face.jpg', 0)

radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(image, n_points, radius, method='uniform')

cv2.imshow('Local Binary Pattern', lbp.astype(np.uint8))
cv2.waitKey()
cv2.destroyAllWindows()