import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

EPOCHS = 50
NUM_CLASSES = 10
BATCH_SIZE = 128
TIME_STEPS = 32  # Number of time steps for RNN input
INPUT_DIM = 32   # Dimension of each time step input


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # normalize
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    return x_train, y_train, x_test, y_test


def build_model():
    model = models.Sequential()

    # RNN layer
    model.add(layers.SimpleRNN(64, input_shape=(TIME_STEPS, INPUT_DIM), return_sequences=True))
    model.add(layers.BatchNormalization())

    # Dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    return model


(x_train, y_train, x_test, y_test) = load_data()

# Reshape input data for RNN
x_train = x_train.reshape((len(x_train), TIME_STEPS, INPUT_DIM))
x_test = x_test.reshape((len(x_test), TIME_STEPS, INPUT_DIM))

model = build_model()
model.compile(loss='categorical_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

# Train
batch_size = 64
model.fit(x_train, y_train, batch_size=batch_size,
          epochs=EPOCHS, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test,
                       batch_size=BATCH_SIZE)

print("\nTest score:", score[0])
print('Test accuracy:', score[1])
