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
model.add(layers.Conv2D(32, (3, 3), activation='relu', \
padding='same', strides = (1,1), input_shape=(28, 28, 1)))
# … to be completed by yourself


# Build the convolutional neural network model



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
    x_train,
    y_train,
    epochs=5,
    batch_size=32
)

print("Training complete.")

# eval test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
y_pred_probs = model.predict(x_test) # predicting, 10k x 10 matrix, 10 prob scores

# pick highest prob in each row, 
# class labels
y_pred = np.argmax(y_pred_probs, axis=1)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion MAtrix:")
print(cm)

# visuals
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classlabels)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - Fashion MNIST")
plt.tight_layout()
plt.show()