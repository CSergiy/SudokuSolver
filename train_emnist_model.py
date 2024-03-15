import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

# Load the EMNIST dataset. 
def load_emnist():
    # Adjust path to where EMNIST data is stored if manually downloaded
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='emnist-byclass.npz')
    
    # Normalize data to range [0, 1]
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Correct orientation of images
    x_train, x_test = x_train.transpose((0, 2, 1)), x_test.transpose((0, 2, 1))
    
    return (x_train, y_train), (x_test, y_test)

# Load data
(x_train, y_train), (x_test, y_test) = load_emnist()

# Define model architecture
model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')  # Number of classes might differ based on EMNIST split
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate model performance
model.evaluate(x_test, y_test)

# Save the model for future use
model.save('my_emnist_model.keras')
