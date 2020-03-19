import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import png

category = ['young_male', 'adult_male', 'old_male', 'young_female', 'adult_female', 'old_female']

imagem = cv2.imread('../material/female/old/KSM0OBKMW1.jpg')
arquivo_modelo = 'model_01_human_category.h5'
arquivo_modelo_json = 'model_01_human_category.json'

json_file = open(arquivo_modelo_json, 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(arquivo_modelo)

original = imagem.copy()
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('../material/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.1, 3)

for (x, y, w, h) in faces:
  cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 1)
  roi_gray = gray[y:y + h, x:x + w]
  roi_gray = roi_gray.astype('float') / 255.0
  cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
  prediction = loaded_model.predict(cropped_img)[0]
  cv2.putText(original, category[int(np.argmax(prediction))], (x, y - 10),
              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('ImageWindow', original)
cv2.waitKey()
# cv2.imwrite("teste_image.png", original)

