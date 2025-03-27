import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import insightface
from insightface.app import FaceAnalysis
# from insightface.data import get_image as ins_get_image
from insightface.model_zoo import get_model




app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))
img = cv2.imread("collage.jpeg")
if img is None:
    raise FileNotFoundError("target2.jpeg not found. Check the file path.")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Matplotlib
fig , axs = plt.subplots(1, 4, figsize=(12, 5))
# plt.imshow(img)
# plt.show()
faces = app.get(img)
# faces = faces[0].keys()
# print(faces)

for i,face  in enumerate(faces):
    bbox = face['bbox']
    bbox = [int(b) for b in bbox]
    axs[i].imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
    axs[i].axis('off')
plt.show()
# print(faces)
swapper = get_model('inswapper_128.onnx', download=False, download_zip=False)

