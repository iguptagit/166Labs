import tensorflow as tf 
from tensorflow import keras 
fashion_mnist = keras.datasets.fashion_mnist 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() 
x_train, x_test = x_train / 255.0, x_test / 255.0 # normalize the pixel values to be in [0, 1] 
# note: you need to flatten the image to a vector, to serve as the input layer of the network. 
# code to be implemented …
print("Train: X = ", x_train.shape)
print("Test: X = ", x_test.shape)

# exploring the data 
classlabels = ['Tshirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-boot']
print(type(x_train))
print(type(y_train))
print(y_train[:10])

# ------------------------------- Part a -------------------------------

# visualizing the dataset
for i in range(10):
    index = np.where(y_train==i)[0][0]
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[index], cmap=plt.get_cmap('gray'))
plt.show()

# ------------------------------- Part b -------------------------------

# Build the neural network model

model = tf.keras.models.Sequential([
    # Flatten 28x28 image into 784 vector
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Hidden layer with 512 neurons and ReLU activation
    tf.keras.layers.Dense(512, activation='relu'),
    # Output layer with 10 neurons and Softmax activation
    tf.keras.layers.Dense(10, activation='softmax')
])

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