import cv2

image = cv2.imread('image.jpeg', 0)

adaptive_thres = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('Adaptve Thresholding', adaptive_thres)
cv2.waitKey()
cv2.destroyAllWindows()