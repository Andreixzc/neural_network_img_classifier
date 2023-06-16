from IPython.display import Image
import tensorflow as tf
import keras as K
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Inicializando a Rede Neural Convolucional
classifier = Sequential()

# Passo 1 - Primeira Camada de Convolução
classifier.add(Conv2D(32, (5, 5), input_shape=(64, 64, 3), activation='relu'))  # Alterando o tamanho do filtro para 5x5

# Adicionando a Segunda Camada de Convolução
classifier.add(Conv2D(64, (3, 3), activation='relu'))  # Alterando o número de filtros para 64

# Adicionando a Terceira Camada de Convolução
classifier.add(Conv2D(128, (3, 3), activation='relu'))  # Alterando o número de filtros para 128

# Adicionando uma Camada de Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adicionando uma Quarta Camada de Convolução
classifier.add(Conv2D(64, (3, 3), activation='relu'))  # Alterando o número de filtros para 64

# Adicionando uma Quinta Camada de Convolução
classifier.add(Conv2D(32, (3, 3), activation='relu'))  # Alterando o número de filtros para 32

# Adicionando uma Segunda Camada de Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Passo 3 - Flattening
classifier.add(Flatten())

# Passo 4 - Full connection
classifier.add(Dense(units=256, activation='relu'))  # Alterando o número de neurônios para 256
classifier.add(Dense(units=128, activation='relu'))  # Alterando o número de neurônios para 128
classifier.add(Dense(units=1, activation='sigmoid'))

# Compilando a rede
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Criando os objetos train_datagen e validation_datagen com as regras de pré-processamento das imagens
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

# Pré-processamento das imagens de treino e validação
print("Training set")
training_set = train_datagen.flow_from_directory('dataset_treino',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


validation_set = validation_datagen.flow_from_directory('dataset_validation',
                                                        target_size = (64, 64),
                                                        batch_size = 32,
                                                        class_mode = 'binary')
# Executando o treinamento (esse processo pode levar bastante tempo, dependendo do seu computador)
classifier.fit_generator(training_set, epochs = 5, validation_data = validation_set,)



classifier.save("modelo.h5")


# Primeira Imagem
import numpy as np
import keras.utils as image

test_image = image.load_img('dataset_teste/2216.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'Cachorro'
else:
    prediction = 'Gato'

print(prediction)