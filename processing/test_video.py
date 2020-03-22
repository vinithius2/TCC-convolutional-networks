import cv2
import os
import numpy as np
import time
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


arquivo_modelo = 'model_01_human_category.h5'
model = load_model(arquivo_modelo)

# Carregando o vídeo
arquivo_video = '../material/teste_02.mp4'
cap = cv2.VideoCapture(arquivo_video)

conectado, video = cap.read()
# print(video.shape) # mostra as dimensões do video

# Redimensionando o tamanho (opcional)
redimensionar = True

largura_maxima = 600

if (redimensionar and video.shape[1]>largura_maxima):
  proporcao = video.shape[1] / video.shape[0]
  video_largura = largura_maxima
  video_altura = int(video_largura / proporcao)
else:
  video_largura = video.shape[1]
  video_altura = video.shape[0]

# Definindo as configurações do vídeo
if not os.path.exists('../material/test_videos'):
    os.makedirs('../material/test_videos')
    print(f'Create directory: material/test_videos')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 24
saida_video = cv2.VideoWriter('../material/test_videos/testado.mp4', fourcc, fps, (video_largura, video_altura))

# Processamento do vídeo e gravação do resultado

# define os tamanhos para as fontes
fonte_pequena, fonte_media = 0.4, 0.7

fonte = cv2.FONT_HERSHEY_SIMPLEX

category = ['young_male', 'adult_male', 'old_male', 'young_female', 'adult_female', 'old_female']

while (cv2.waitKey(1) < 0):
    conectado, frame = cap.read()
    if not conectado:
        break
    t = time.time()
    if redimensionar:
        frame = cv2.resize(frame, (video_largura, video_altura))
    face_cascade = cv2.CascadeClassifier('../material/haarcascade_frontalface_default.xml')
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(cinza, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h + 10), (255, 50, 50), 2)
            roi = cinza[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            result = model.predict(roi)[0]
            print(result)
            if result is not None:
                resultado = np.argmax(result)
                cv2.putText(frame, category[resultado], (x, y - 10), fonte, fonte_media, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, " frame processado em {:.2f} segundos".format(time.time() - t), (20, video_altura - 20), fonte,
                fonte_pequena, (250, 250, 250), 0, lineType=cv2.LINE_AA)

    saida_video.write(frame)

print("Terminou")
saida_video.release()
