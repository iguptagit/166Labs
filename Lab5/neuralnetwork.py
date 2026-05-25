import tensorflow as tf 
from tensorflow import keras 
fashion_mnist = keras.datasets.fashion_mnist 
import matplotlib.pyplot as plt
import numpy as np

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

# visualizing the dataset
for i in range(10):
    index = np.where(y_train==i)[0][0]
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[index], cmap=plt.get_cmap('gray'))
plt.show()


