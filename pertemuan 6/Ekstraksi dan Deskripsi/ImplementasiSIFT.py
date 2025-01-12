import cv2

image = cv2.imread('image.jpeg')

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(image, None)

sift_image = cv2.drawKeypoints(image, keypoints, None)

cv2.imshow('SIFT features', sift_image)
cv2.waitKey()
cv2.destroyAllWindows()
