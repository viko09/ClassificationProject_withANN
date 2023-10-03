from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout

import mlflow

# Importamos los datos de entrenamiento y prueba de nuestra red
train_dir = '/home/vikoluna/NeuralCatDog/data/train'
test_dir = '/home/vikoluna/NeuralCatDog/data/test'

# Esto es por perros y gatos (0 y 1 o sea seran 2 clases)
num_class = 2
# Es el número de épocas
epoch = 10

# Numero de imagenes
num_train = 10001
num_test = 2501

# Como tenemos imagenes de distintos tamaños, vamos a ajustar las imagenes a nuevas dimensiones
ih, iw = 150, 150
# Definimos la forma, alto, ancho y canales RGB
input_shape = (ih, iw, 3)

# Definimos el tamaño del batch
batch_size = 20

epoch_steps = num_train // batch_size
test_steps = num_test // batch_size

# Indica que reescale cada canal con valor entre 0 y 1
gentrain = ImageDataGenerator(rescale=1. / 255.)

train = gentrain.flow_from_directory(train_dir,
                                     batch_size=batch_size,
                                     target_size=(iw, ih),
                                     class_mode='binary')

gentest = ImageDataGenerator(rescale=1. / 255.)

test = gentest.flow_from_directory(train_dir,
                                     batch_size=batch_size,
                                     target_size=(iw, ih),
                                     class_mode='binary')

# Comenzamos a crear nuestra arquitectura de red secuencial
model = Sequential()

# Comenzaremos agregando una capa de entrada
model.add(Conv2D(10, (5, 5), input_shape=(ih, iw, 3)))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capas ocultas
model.add(Conv2D(10, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Capa de salida
# Aplanamos la imagen
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('softmax'))
# Regularización
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compilamos el modelo
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


with mlflow.start_run() as run:
    model.fit_generator(
        train,
        steps_per_epoch=epoch_steps,
        epochs=epoch,
        validation_data=test,
        validation_steps=test_steps
    )

# Salvamos el modelo
# model.save()
