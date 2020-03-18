import cv2
import os
import csv
import zipfile

cascade_faces = '../material/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(cascade_faces)

path = "../material/faces.zip"
zip_object = zipfile.ZipFile(file=path, mode="r")
zip_object.extractall("../material")

male_list = os.listdir('../material/male/')
female_list = os.listdir('../material/female/')
category = {'male': male_list, 'female': female_list}
to_list = [['index', 'category', 'pixels']]
category_cvs = {
    'young_male': 0,
    'adult_male': 1,
    'old_male': 2,
    'young_female': 3,
    'adult_female': 4,
    'old_female': 5,
}
index = 0
"""
    Pega-se todas as imagens comuns com RGB e alta resolução e transforma em imagens de tonalização cinza por 48 pixels
    de altura e largura. 
    
"""
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

                all_pixels = str()
                all_array_pixels = []
                for pixels in roi:
                    all_array_pixels.append(" ".join(str(x) for x in pixels))
                all_pixels = " ".join(str(x) for x in all_array_pixels)
                index += 1
                to_list.append([index, category_cvs[f'{name}_{key}'], all_pixels])

with open('human_category.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(to_list)
