import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

male_list = os.listdir('../material/image_grey/male')
female_list = os.listdir('../material/image_grey/female')
category = {'male': male_list, 'female': female_list}

"""
    Converter as imagens cinzas no formato que o TensorFlow reconheça.
"""
data = pd.read_csv('./category_human.csv')
print(data.tail())

# plt.figure(figsize=(12, 6))
# plt.hist(data['category'], bins=30)
# plt.title('Imagens x Categorias')
# Category = ['young_male', 'adult_male', 'old_male', 'young_female', 'adult_female', 'old_female']

pixels = data['pixels'].tolist()
largura, altura = 48, 48
faces = []
amostras = 0

for pixel_sequence in pixels:
  face = [int(pixel) for pixel in pixel_sequence.split(' ')]
  face = np.asarray(face).reshape(largura, altura)
  faces.append(face)
  amostras += 1

# print('Número total de imagens no dataset: ', str(len(faces)))
faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)
faces = faces.astype('float32')
faces = faces / 255.0
category = pd.get_dummies(data['category']).values
# print('Category: ', category[0])
# print('FACES: ', faces.shape)
# print('FACES First: ', faces[0])

"""
    Base de treinamento, teste e validação
"""
# Base treinamento
X_train, X_test, y_train, y_test = train_test_split(faces, category, test_size=0.1, random_state=42)
# Base de validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)
# print('Número de imagens no conjunto de treinamento:', len(X_train))
# print('Número de imagens no conjunto de teste:', len(X_test))
# print('Número de imagens no conjunto de validação:', len(X_val))
np.save('mod_xtest', X_test)
np.save('mod_ytest', y_test)

"""
    Criação das Redes Neurais
"""
num_features = 64
num_labels = 6
batch_size = 64
epochs = 100
width, height = 48, 48

model = Sequential()

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu',
                 input_shape=(width, height, 1), data_format = 'channels_last',
                 kernel_regularizer = l2(0.01)))

# Camada de convolução e Pooling
model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

# Flattening
model.add(Flatten())

# Rede neural
model.add(Dense(2*2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*num_features, activation='relu'))
model.add(Dropout(0.5))
# Saída rede neural
model.add(Dense(num_labels, activation='softmax'))
# model.summary()

"""
    Copilando modelo
"""
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr = 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

arquivo_modelo = 'model_01_human_category.h5'
arquivo_modelo_json = 'model_01_human_category.json'

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(arquivo_modelo, monitor='val_loss', verbose=1, save_best_only=True)

"""
    Salvando a arquitetura do modelo em um arquivo JSON
"""
model_json = model.to_json()
with open(arquivo_modelo_json, 'w') as json_file:
  json_file.write(model_json)

"""
    Treinando o modelo
"""
history = model.fit(np.array(X_train), np.array(y_train),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(np.array(X_val), np.array(y_val)),
                    shuffle=True,
                    callbacks=[lr_reducer, early_stopper, checkpointer])
