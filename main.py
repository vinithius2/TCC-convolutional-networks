import getopt
import sys
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


def main(argv):
    """
        Função principal que executa todos passos por linha de comando, desde a criação do dataset aos teste de
        detectação facial em vídeo ou imagem.
        :param argv: list()
    """
    try:
        opts, args = getopt.getopt(argv, "i:v:dtrh", ["image=", "video=", "dataset", "training", "real", "help"])
    except getopt.GetoptError as e:
        print("Erro: ", e, "\n")
        help()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--dataset"):
            create_dataset()
        elif opt in ("-t", "--training"):
            training()
        elif opt in ("-i", "--image"):
            if not os.path.exists(arg):
                print('Caminho desconhecido, tente novamente.')
            else:
                detect_face_in_image(arg)
        elif opt in ("-v", "--video"):
            if not os.path.exists(arg):
                print('Caminho desconhecido, tente novamente.')
            else:
                detect_face_in_video(arg)
        elif opt in ("-r", "--real"):
            print('# REAL')
            detect_face_in_realtime()
        elif opt in ("-h", "--help"):
            help()
        else:
            print('# Argumento inválido, segue dicas: .\n')
            help()
            sys.exit()


def help():
    """
        Dicas de comandos para o terminal
        :return:
    """
    print('\n# Descompacta as imagens e cria o dataset para rede neural.')
    print('main.py -d --dataset\n')
    print('# Treina e testar a rede neural, gerando gráficos para o entendimento do treinamento.')
    print('main.py -t --training\n')
    print('# Inicia a dectação na imagem passada pelo PATH')
    print('main.py -i <path> --image <path>\n')
    print('# Inicia a dectação no vídeo passado pelo PATH')
    print('main.py -v <path> --video <path>\n')
    print('# Inicia a dectação em tempo real pela webcam')
    print('main.py -r <path> --real <path>\n')


def create_dataset():
    """
        Descompactar imagens e criar dataset para rede neural.
    """
    extrat_zip()
    image_processing()


def training():
    """
        Treinar e testar a rede neural, gerar gráficos para o entendimento do treinamento.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = test_base_validation()
    faces, category = convert_images_for_tensorflow()
    model = create_neural_network()
    lr_reducer, early_stopper, checkpointer = model_compile(model)
    save_json(model)
    history = model_training(model, X_train, y_train, X_val, y_val, lr_reducer, early_stopper, checkpointer)
    create_graph_accuracy(history)
    scores = checking_model_accuracy(model, X_test, y_test)
    data_to_generate_the_confusion_matrix()


def detect_face_in_image(path):
    """
        Detecção facial por imagem.
        :return:
    """
    import numpy as np
    from tensorflow.keras.models import model_from_json

    category = ['young_male', 'adult_male', 'old_male', 'young_female', 'adult_female', 'old_female']

    imagem = cv2.imread(path)

    name_file = path.split('/')
    name_file = name_file[-1:][0]
    name_file = name_file.split('.')
    name_file = name_file[0]

    arquivo_modelo = 'processing/model_01_human_category.h5'
    arquivo_modelo_json = 'processing/model_01_human_category.json'

    json_file = open(arquivo_modelo_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(arquivo_modelo)

    original = imagem.copy()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier('material/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    for (x, y, w, h) in faces:
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 1)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = roi_gray.astype('float') / 255.0
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = loaded_model.predict(cropped_img)[0]
        cv2.putText(original, category[int(np.argmax(prediction))], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    if not os.path.exists('material/test_images'):
        os.makedirs('material/test_images')
        print(f'Create directory: material/test_images')
    cv2.imwrite(f'material/test_images/{name_file}.png', original)


def detect_face_in_video(path):
    """
        Detecção facial por vídeo
        :return:
    """
    import numpy as np
    import time
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import load_model

    arquivo_modelo = 'processing/model_01_human_category.h5'
    model = load_model(arquivo_modelo)

    cap = cv2.VideoCapture(path)

    name_file = path.split('/')
    name_file = name_file[-1:][0]
    name_file = name_file.split('.')
    name_file = name_file[0]

    conectado, video = cap.read()

    redimensionar = True
    largura_maxima = 600

    if redimensionar and video.shape[1] > largura_maxima:
        proporcao = video.shape[1] / video.shape[0]
        video_largura = largura_maxima
        video_altura = int(video_largura / proporcao)
    else:
        video_largura = video.shape[1]
        video_altura = video.shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24
    if not os.path.exists('material/test_videos'):
        os.makedirs('material/test_videos')
        print(f'Create directory: material/test_videos')
    saida_video = cv2.VideoWriter(f'material/test_videos/{name_file}.mp4', fourcc, fps, (video_largura, video_altura))

    fonte_pequena, fonte_media = 0.4, 0.7
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    category = ['young_male', 'adult_male', 'old_male', 'young_female', 'adult_female', 'old_female']

    while cv2.waitKey(1) < 0:
        conectado, frame = cap.read()
        if not conectado:
            break
        t = time.time()
        if redimensionar:
            frame = cv2.resize(frame, (video_largura, video_altura))
        face_cascade = cv2.CascadeClassifier('material/haarcascade_frontalface_default.xml')
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
                    cv2.putText(frame, category[resultado], (x, y - 10), fonte, fonte_media, (255, 255, 255), 1,
                                cv2.LINE_AA)

        cv2.putText(frame, " frame processado em {:.2f} segundos".format(time.time() - t), (20, video_altura - 20),
                    fonte,
                    fonte_pequena, (250, 250, 250), 0, lineType=cv2.LINE_AA)

        saida_video.write(frame)

    print("Terminou")
    saida_video.release()


def detect_face_in_realtime():
    """
        Detecção facial em tempo real pela webcam
        :return:
    """
    import time
    import matlab.engine
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import load_model

    arquivo_modelo = 'processing/model_01_human_category.h5'
    model = load_model(arquivo_modelo)

    cap = cv2.VideoCapture(0)

    conectado, video = cap.read()

    redimensionar = True
    largura_maxima = 600

    if redimensionar and video.shape[1] > largura_maxima:
        proporcao = video.shape[1] / video.shape[0]
        video_largura = largura_maxima
        video_altura = int(video_largura / proporcao)
    else:
        video_largura = video.shape[1]
        video_altura = video.shape[0]

    fonte_pequena, fonte_media = 0.4, 0.7
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    category = ['young_male', 'adult_male', 'old_male', 'young_female', 'adult_female', 'old_female']

    while cv2.waitKey(1) < 0:
        conectado, frame = cap.read()
        if not conectado:
            break
        t = time.time()
        if redimensionar:
            frame = cv2.resize(frame, (video_largura, video_altura))
        face_cascade = cv2.CascadeClassifier('material/haarcascade_frontalface_default.xml')
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
                    # image_01 = cv2.imread('material/01_realtime.png')
                    # if image_01:
                    #     image_02 = cv2.imread(frame)
                    #     I = np.single(rgb2gray(I))
                    #     J = single(rgb2gray(J))
                    # cv2.imwrite('material/01_realtime.png', frame)
                    resultado = np.argmax(result)
                    cv2.putText(frame, category[resultado], (x, y - 10), fonte, fonte_media, (255, 255, 255), 1,
                                cv2.LINE_AA)

        cv2.putText(frame,
                    " frame processado em {:.2f} segundos".format(time.time() - t),
                    (20, video_altura - 20),
                    fonte,
                    fonte_pequena,
                    (250, 250, 250),
                    0,
                    lineType=cv2.LINE_AA)

        cv2.imshow('object detection', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def test_base_validation(faces, category):
    """
        Base de treinamento, teste e validação
        :param faces:
        :param category:
        :return:
    """
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Base treinamento
    X_train, X_test, y_train, y_test = train_test_split(faces, category, test_size=0.1, random_state=42)
    # Base de validação
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)
    print('Número de imagens no conjunto de treinamento:', len(X_train))
    print('Número de imagens no conjunto de teste:', len(X_test))
    print('Número de imagens no conjunto de validação:', len(X_val))
    np.save('material/mod_xtest', X_test)
    np.save('material/mod_ytest', y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def convert_images_for_tensorflow():
    """
        Converter as imagens cinzas no formato que o TensorFlow reconheça.
        :return:
    """
    import numpy as np
    import pandas as pd

    data = pd.read_csv('material/category_human.csv')
    print(data.tail())

    pixels = data['pixels'].tolist()
    largura, altura = 48, 48
    faces = []
    amostras = 0

    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(largura, altura)
        faces.append(face)
        amostras += 1

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    faces = faces.astype('float32')
    faces = faces / 255.0
    category = pd.get_dummies(data['category']).values
    print('Número total de imagens no dataset: ', str(len(faces)))
    return faces, category


def create_neural_network():
    """
        Criação das Redes Neurais
        :return:
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
    from tensorflow.keras.regularizers import l2

    num_features = 64
    num_labels = 6
    width, height = 48, 48
    model = Sequential()
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu',
                     input_shape=(width, height, 1), data_format='channels_last',
                     kernel_regularizer=l2(0.01)))

    # Camada de convolução e Pooling
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    # Flattening
    model.add(Flatten())
    # Rede neural
    model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * num_features, activation='relu'))
    model.add(Dropout(0.5))
    # Saída rede neural
    model.add(Dense(num_labels, activation='softmax'))
    # model.summary()
    return model


def model_compile(model):
    """
        Copilando modelo
        :param model:
        :return:
    """
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    arquivo_modelo = 'model_01_human_category.h5'

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
    checkpointer = ModelCheckpoint(arquivo_modelo, monitor='val_loss', verbose=1, save_best_only=True)
    return lr_reducer, early_stopper, checkpointer


def save_json(model):
    """
        Salvando a arquitetura do modelo em um arquivo JSON
        :param model:
        :return:
    """
    arquivo_modelo_json = 'model_01_human_category.json'
    model_json = model.to_json()
    with open(arquivo_modelo_json, 'w') as json_file:
        json_file.write(model_json)


def model_training(model, X_train, y_train, X_val, y_val, lr_reducer, early_stopper, checkpointer):
    """
        Treinando o modelo
        :param model:
        :param X_train:
        :param y_train:
        :param X_val:
        :param y_val:
        :param lr_reducer:
        :param early_stopper:
        :param checkpointer:
        :return:
    """
    import numpy as np

    batch_size = 64
    epochs = 100
    history = model.fit(np.array(X_train), np.array(y_train),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(np.array(X_val), np.array(y_val)),
                        shuffle=True,
                        callbacks=[lr_reducer, early_stopper, checkpointer])
    return history


def create_graph_accuracy(history):
    """
        Gerando os gráficos
        :param history:
        :return:
    """

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], 'r')
    axs[0].plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], 'b')
    axs[0].set_title('Acurácia do modelo')
    axs[0].set_ylabel('Acurácia')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(history.history['accuracy']) + 1),
                      len(history.history['accuracy']) / 10)
    axs[0].legend(['training accuracy', 'validation accuracy'], loc='best')

    axs[1].plot(range(1, len(history.history['loss']) + 1), history.history['loss'], 'r')
    axs[1].plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'], 'b')
    axs[1].set_title('Loss do modelo')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(history.history['loss']) + 1),
                      len(history.history['loss']) / 10)
    axs[1].legend(['training loss', 'validation loss'], loc='best')
    fig.savefig('material/history_mod01.png')


def data_to_generate_the_confusion_matrix():
    """
        Gerando os dados para a geração da matriz de confusão
        :return:
    """
    from tensorflow.keras.models import model_from_json

    true_y = []
    pred_y = []
    arquivo_modelo_json = 'model_01_human_category.json'
    arquivo_modelo = 'model_01_human_category.h5'
    x = np.load('processing/mod_xtest.npy')
    y = np.load('processing/mod_ytest.npy')
    json_file = open(arquivo_modelo_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(arquivo_modelo)
    y_pred = loaded_model.predict(x)
    yp = y_pred.tolist()
    yt = y.tolist()
    count = 0
    for i in range(len(y)):
        yy = max(yp[i])
        yyt = max(yt[i])
        pred_y.append(yp[i].index(yy))
        true_y.append(yt[i].index(yyt))
        if yp[i].index(yy) == yt[i].index(yyt):
            count += 1
    acc = (count / len(y)) * 100
    print('Acurácia no conjunto de teste: ' + str(acc))
    np.save('material/truey_mod01', true_y)
    np.save('material/predy_mod01', pred_y)


def checking_model_accuracy(model, X_test, y_test):
    """
        Conferindo a acurácia do modelo
        :param model:
        :param X_test:
        :param y_test:
        :return:
    """
    batch_size = 64
    scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
    print('Acurácia: ' + str(scores[1]))
    print('Erro: ' + str(scores[0]))
    return scores


def generate_the_confusion_matrix():
    """
        Gerando a Matriz de Confusão
        :return:
    """
    from sklearn.metrics import confusion_matrix
    import itertools

    y_true = np.load('material/truey_mod01.npy')
    y_pred = np.load('material/predy_mod01.npy')
    cm = confusion_matrix(y_true, y_pred)
    category = ['young_male', 'adult_male', 'old_male', 'young_female', 'adult_female', 'old_female']
    titulo = 'Matriz de Confusão'
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(titulo)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('Classificação correta')
    plt.xlabel('Predição')
    plt.savefig('material/matriz_confusao_mod01.png')


def extrat_zip():
    """
        Extração das imagens base
        :return:
    """
    import zipfile

    path = "material/image_grey.zip"
    zip_object = zipfile.ZipFile(file=path, mode="r")
    zip_object.extractall("material")


def image_processing():
    """
        Pega-se todas as imagens comuns com RGB e alta resolução e transforma em imagens de tonalização cinza por
        48 pixels de altura e largura.
        :return:
    """
    import csv

    cascade_faces = 'material/haarcascade_frontalface_default.xml'
    face_detection = cv2.CascadeClassifier(cascade_faces)
    male_list = os.listdir('material/male/')
    female_list = os.listdir('material/female/')
    category = {'male': male_list, 'female': female_list}
    to_list = [['index', 'category', 'pixels']]
    index = 0
    category_cvs = {
        'young_male': 0,
        'adult_male': 1,
        'old_male': 2,
        'young_female': 3,
        'adult_female': 4,
        'old_female': 5,
    }
    for key, list in category.items():
        for name in list:
            files = os.listdir(f'material/{key}/{name}')
            for file in files:
                image = cv2.imread(f'material/{key}/{name}/{file}')
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

    with open('material/human_category.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(to_list)


if __name__ == "__main__":
    main(sys.argv[1:])
