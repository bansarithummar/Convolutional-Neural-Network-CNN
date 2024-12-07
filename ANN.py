import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# Sample data
X_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [[0], [1], [1], [0]]

# Define the model
model = Sequential([
    Dense(2, input_shape=(2,), activation='relu'),  # Input layer with 2 neurons
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=1000, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X_train, y_train)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Make prediction
X_test = [[0, 0], [0, 1], [1, 0], [1, 1]]
predictions = model.predict(X_test)
print("Predictions:")
for i, x in enumerate(X_test):
    print(f"Input: {x}, Predicted Output: {predictions[i]}")
