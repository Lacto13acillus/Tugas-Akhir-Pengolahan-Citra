import cv2
import numpy as np

# Membaca gambar
image_path = "image.jpeg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Inisialisasi ORB
orb = cv2.ORB_create()

# Deteksi dan deskripsi fitur
keypoints, descriptors = orb.detectAndCompute(img, None)

# Gambar keypoints pada gambar
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

# Tampilkan gambar dengan keypoints
cv2.imshow("ORB Keypoints", img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
