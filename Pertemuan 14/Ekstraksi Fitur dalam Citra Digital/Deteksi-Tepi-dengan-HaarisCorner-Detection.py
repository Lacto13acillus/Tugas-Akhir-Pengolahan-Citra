import cv2
import numpy as np

# Load gambar grayscale
image = cv2.imread('cp10.jpeg', cv2.IMREAD_GRAYSCALE)

# Konversi gambar grayscale ke RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Deteksi sudut
corners = cv2.cornerHarris(image, 2, 3, 0.04)

# Tandai sudut pada gambar RGB
image_rgb[corners > 0.01 * corners.max()] = [0, 0, 255]

# Tampilkan hasil
cv2.imshow('Harris Corner Detection', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
