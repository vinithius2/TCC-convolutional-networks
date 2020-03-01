# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
cascade_faces = 'material/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(cascade_faces)

adulto_list = os.listdir('material/adulto/')
crianca_list = os.listdir('material/crianca/')
velho_list = os.listdir('material/velho/')
categorias = {'adulto': adulto_list, 'crianca': crianca_list, 'velho': velho_list}

for key, list in categorias.items():
    for name in list:
        print(key, name)
        imagem = cv2.imread(f'material/{key}/{name}')
        original = imagem.copy()
        faces = face_detection.detectMultiScale(original, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        cinza = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        for idx, face in enumerate(faces):
            # Pegando informações da captura de face
            x = face[0]
            y = face[1]
            boxe_x = face[2]
            boxe_y = face[3]
            # Capturando somente o rosto de redimensionando
            roi = cinza[y:y + boxe_y, x:x + boxe_x]
            roi = cv2.resize(roi, (48, 48))
            if not os.path.exists(f'material/{key}/gray'):
                os.mkdir(f'material/{key}/gray')
                print('Criou pasta')
            cv2.imwrite(f'material/{key}/gray/{name}_{idx}.jpg', roi)

            # Preparando imagem para o tensorflow
            # roi = roi.astype('float')
            # roi = roi / 255
            # roi = img_to_array(roi)
            # roi = np.expand_dims(roi, axis=0)
            # cv2.imshow('', roi)
            # print(roi.shape)