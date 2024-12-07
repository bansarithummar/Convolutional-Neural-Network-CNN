import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_rnn_model(input_shape, num_classes):
    # CNN part
    cnn_input = layers.Input(shape=input_shape)
    cnn_conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(cnn_input)
    cnn_maxpool1 = layers.MaxPooling2D(pool_size=(2, 2))(cnn_conv1)
    cnn_dropout1 = layers.Dropout(0.25)(cnn_maxpool1)
    cnn_flatten = layers.Flatten()(cnn_dropout1)

    # RNN part
    rnn_input = layers.Input(shape=(None, input_shape[0]))  # Assuming input_shape[0] is the time step dimension
    rnn_lstm1 = layers.LSTM(64, return_sequences=True)(rnn_input)
    rnn_lstm2 = layers.LSTM(64)(rnn_lstm1)

    # Concatenate CNN and RNN outputs
    combined = layers.concatenate([cnn_flatten, rnn_lstm2])

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(combined)
    dropout2 = layers.Dropout(0.5)(dense1)
    output = layers.Dense(num_classes, activation='softmax')(dropout2)

    # Define model
    model = models.Model(inputs=[cnn_input, rnn_input], outputs=output)
    return model

# Example usage
input_shape = (28, 28, 1)  # Example input shape for CNN
time_steps = 10  # Example number of time steps for RNN
num_classes = 10  # Example number of output classes
model = build_cnn_rnn_model(input_shape, num_classes)
model.summary()
