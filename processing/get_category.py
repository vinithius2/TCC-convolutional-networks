import cv2
import os
import numpy as np
from tensorflow.keras.models import model_from_json

category = ['young_male', 'adult_male', 'old_male', 'young_female', 'adult_female', 'old_female']
face_cascade = cv2.CascadeClassifier('../material/haarcascade_frontalface_default.xml')
arquivo_modelo = 'model_01_human_category.h5'
arquivo_modelo_json = 'model_01_human_category.json'
json_file = open(arquivo_modelo_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(arquivo_modelo)
path_dir = '/home/vinithius/Downloads/kaggle/1-million-fake-faces'
diretorios_01 = os.listdir(path_dir)
count = 0
for dir_01 in diretorios_01:
    diretorios_02 = os.listdir(f'{path_dir}/{dir_01}')
    for dir_02 in diretorios_02:
        diretorios_03 = os.listdir(f'{path_dir}/{dir_01}/{dir_02}')
        for dir_03 in diretorios_03:
            files = os.listdir(f'{path_dir}/{dir_01}/{dir_02}/{dir_03}')
            for file in files:
                imagem = cv2.imread(f'{path_dir}/{dir_01}/{dir_02}/{dir_03}/{file}')
                original = imagem.copy()
                gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 3)
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = roi_gray.astype('float') / 255.0
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    prediction = loaded_model.predict(cropped_img)[0]
                    type = category[int(np.argmax(prediction))]
                    if not os.path.exists(f'../material/get_images/{type}'):
                        os.makedirs(f'../material/get_images/{type}')
                        print(f'\nCreate directory: ../material/get_images/{type}\n')
                    cv2.imwrite(f'../material/get_images/{type}/{file}', original)
                    count += 1
                    print(f'{count} processadas...')
