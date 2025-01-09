import cv2

image = cv2.imread('image.jpeg', 0)

edges = cv2.Canny(image, 100, 200)

cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey()
cv2.destroyAllWindows()

