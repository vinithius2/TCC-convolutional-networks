# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
# Helper libraries
import numpy as np
import zipfile
import matplotlib.pyplot as plt
cascade_faces = '../material/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(cascade_faces)

path = "../material/faces.zip"
zip_object = zipfile.ZipFile(file=path, mode="r")
zip_object.extractall("../material")

male_list = os.listdir('../material/male/')
female_list = os.listdir('../material/female/')

category = {'male': male_list, 'female': female_list}

for key, list in category.items():
    for name in list:
        files = os.listdir(f'../material/{key}/{name}')
        for file in files:
            image = cv2.imread(f'../material/{key}/{name}/{file}')
            original = image.copy()
            faces = face_detection.detectMultiScale(original, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            grey = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            for idx, face in enumerate(faces):
                # Pegando informações da captura de face
                x = face[0]
                y = face[1]
                boxe_x = face[2]
                boxe_y = face[3]
                # Capturando somente o rosto de redimensionando
                roi = grey[y:y + boxe_y, x:x + boxe_x]
                roi = cv2.resize(roi, (48, 48))

                if not os.path.exists(f'material/image_grey/{key}/{name}'):
                    os.makedirs(f'material/image_grey/{key}/{name}')
                    print(f'Create directory: material/image_grey/{key}/{name}')
                cv2.imwrite(f'material/image_grey/{key}/{name}/{file}.jpg', roi)
