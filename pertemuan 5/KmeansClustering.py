import cv2
import numpy as np

image = cv2.imread('image.jpeg')
Z = image.reshape((-1, 3))

Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
segmented_image = res.reshape((image.shape))

cv2.imshow('kmeans segmentation', segmented_image)
cv2.waitKey()
cv2.destroyAllWindows()