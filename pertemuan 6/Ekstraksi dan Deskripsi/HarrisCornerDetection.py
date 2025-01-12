import cv2
import numpy as np

image = cv2.imread('image.jpeg', 0)

gray = np.float32(image)

dst = cv2.cornerHarris(gray, 2, 3, 0.04)

dst = cv2.dilate(dst, None)

color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

color_image[dst > 0.01 * dst.max()] = [0, 0, 255]


cv2.imshow('Harris Corners', image)
cv2.waitKey()
cv2.destroyAllWindows()
