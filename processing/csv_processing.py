import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

male_list = os.listdir('../material/image_grey/male')
female_list = os.listdir('../material/image_grey/female')

category = {'male': male_list, 'female': female_list}

for key, list in category.items():
    for name in list:
        files = os.listdir(f'../material/image_grey/{key}/{name}')
        for file in files:
            image = cv2.imread(f'../material/image_grey/{key}/{name}/{file}')
            roi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            roi = roi.astype('float')
            roi = roi / 255
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            print(roi.shape)
            # cv2.imshow('', roi)
            # print(roi.shape)