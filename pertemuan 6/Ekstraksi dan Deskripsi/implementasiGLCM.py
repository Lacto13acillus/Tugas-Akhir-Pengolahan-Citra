from skimage.feature import graycomatrix, graycoprops
import cv2

image = cv2.imread('image.jpeg', 0)

glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

contrast = graycoprops(glcm, 'contrast')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]

print(f'contrast: {contrast}, energy: {energy}')

