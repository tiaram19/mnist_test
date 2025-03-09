import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Membuat model CNN sederhana
def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# Load dataset MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalisasi
x_train = x_train[..., tf.newaxis]  # Tambah dimensi channel
x_test = x_test[..., tf.newaxis]

# Buat model
model = build_model()

# Train model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Simpan model
model.save("model_cnn.h5")