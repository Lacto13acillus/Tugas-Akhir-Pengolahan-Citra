import cv2
import numpy as np

image = cv2.imread('image.jpeg')

ret, thresh_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('tresholded image', thresh_image)
cv2.waitKey()
cv2.destroyAllWindows()