import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
fashion_mnist = keras.datasets.fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()

# Layer 2 (provided)
model.add(layers.Conv2D(
    32,
    (3, 3),
    activation='relu',
    padding='same',
    strides=(1,1),
    input_shape=(28, 28, 1)
))

# Layer 3
model.add(layers.MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2),
    padding='same'
))

# Layer 4
model.add(layers.Conv2D(
    64,
    (3,3),
    activation='relu',
    padding='same',
    strides=(1,1)
))

# Layer 5
model.add(layers.MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2),
    padding='same'
))

# Layer 6
model.add(layers.Conv2D(
    64,
    (3,3),
    activation='relu',
    padding='same',
    strides=(1,1)
))

# Layer 7
model.add(layers.Flatten())

# Build the convolutional neural network model
# Remaining layers to be added 

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train the model
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=32
)

print("Training complete.")

# Evaluate test set
test_loss, test_accuracy = model.evaluate(
    test_images,
    test_labels
)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

y_pred_probs = model.predict(test_images)

# Pick highest probability class
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(test_labels, y_pred)

print("Confusion Matrix:")
print(cm)

classlabels = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classlabels
)

disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - Fashion MNIST")
plt.tight_layout()
plt.show()