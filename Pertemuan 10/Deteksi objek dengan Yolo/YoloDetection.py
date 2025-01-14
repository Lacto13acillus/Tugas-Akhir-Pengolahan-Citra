import cv2
import numpy as np

#Load YOLO model dan konfigurasi
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#Baca Citra
image = cv2.imread('cp10.jpeg')
height, width, channels = image.shape

#Persiapkan Input YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

#Loop melalui deteksi gambar dan Bounding Box
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            #Ekstrak koordinat Bounding Box
            center_x = int(detection[0] * width)
            center_y = int(detection[0] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            #Gambar kotak pada objek yang terdeteksi
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

#Menampilkan Hasil
cv2.imshow("YOLO detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
