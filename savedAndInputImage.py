import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Python Imaging Library (expansion of PIL) is the de facto image processing 
# package.
from PIL import Image, ImageOps

# Prepare the dataset
mnist = keras.datasets.fashion_mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
class_names = ['top', 'trouser', 'pullover', 'dress', 'coat', \
    'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']

# load a Jason model file
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights of a model via a H5 file
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# You need to re-compile the loaded model before use for efficient computation 
# But, you don't need to train the model again
loaded_model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', \
    metrics = ['accuracy'])

# You may evaluate it 
test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
print("Evaluation accuracy = ", 100*test_acc, "%")

# Make prediction on test data
pred = loaded_model.predict(x_test) 
# pred contains all the prediction results
# pred[0] refers to the prediction result of the first test image: x_test[0]
print("Probabilities of each category for x_test[0] image: \n")
print(pred[0], '\n')
print('Shape of pred[0]: ', pred[0].shape, '\n')
print("Predicted category for x_test[0] image: \n")
print(class_names[np.argmax(pred[0])]) # np.argmax(a_list or an_array) returns 
# the index of the maximum item in a list or the indices of the maximum elements
# in an array along one axis
print("Display x_test[0] image: \n")
plt.figure()
plt.imshow(x_test[0])  # a fake image controlled by the color bar
plt.colorbar()
plt.show()

# convert a numpy array to an image
true_image = Image.fromarray(x_test[0])
true_image.show()

print("Probabilities of each category for x_test[1] image: \n")
print(pred[1], '\n')
print('Shape of pred[1]: ', pred[1].shape, '\n')
print("Predicted category for x_test[1] image: \n")
print(class_names[np.argmax(pred[1])]) # np.argmax(a_list or an_array) returns 
# the index of the maximum item in a list or the indices of the maximum elements
# in an array along one axis
print("Display x_test[1] image: \n")
plt.figure()
plt.imshow(x_test[1])
plt.colorbar()
plt.show()

 # changed here, was not showing the mini image like the shoe did, was missing this statement
true_image = Image.fromarray(x_test[1])
true_image.show()



# test an input image: bag
path_bag = r'C:\Users\airik\Desktop\bag.png'
img_bag = Image.open(path_bag)  # in RGB format
img_bag1 = ImageOps.grayscale(img_bag)
img_bag2 = img_bag1.resize((28,28))
#img_bag.show()
img_bag2.show()
img_bag3 = np.expand_dims(img_bag2, axis=0)

pred2 = loaded_model.predict(img_bag3) 
#img_bag4 = np.expand_dims(x_test[1], axis=0)
#pred2 = loaded_model.predict(img_bag4)
print("Probabilities of each category for imag_bag3: \n")
print(pred2[0], '\n')
print('Shape of pred2[0]: ', pred2[0].shape, '\n')
print("Predicted category for x_test[0] image: \n")
print(class_names[np.argmax(pred2[0])]) # np.argmax(a_list or an_array) returns 
# the index of the maximum item in a list or the indices of the maximum elements
# in an array along one axis
print("Display img_bag3[0]: \n")
plt.figure()
plt.imshow(img_bag3[0])
plt.colorbar()
plt.show()


# MY test an input image: dress
path_dress = r'C:\Users\airik\Desktop\dress.jpg'
img_dress = Image.open(path_dress)  # in RGB format
img_dress1 = ImageOps.grayscale(img_dress)
img_dress2 = img_dress1.resize((28,28))
#img_dress.show()
img_dress2.show()
img_dress3 = np.expand_dims(img_dress2, axis=0)

predDress2 = loaded_model.predict(img_dress3) 
#img_dress4 = np.expand_dims(x_test[1], axis=0)
#pred2 = loaded_model.predict(img_dress4)
print("Probabilities of each category for imag_dress3: \n")
print(predDress2[0], '\n')
print('Shape of predDress2[0]: ', predDress2[0].shape, '\n')
print("Predicted category for x_test[0] image: \n")
print(class_names[np.argmax(predDress2[0])]) # np.argmax(a_list or an_array) returns 
# the index of the maximum item in a list or the indices of the maximum elements
# in an array along one axis
print("Display img_dress3[0]: \n")
plt.figure()
plt.imshow(img_dress3[0])
plt.colorbar()
plt.show()



# MY test an input image: coat
path_coat = r'C:\Users\airik\Desktop\coat.jpg'
img_coat = Image.open(path_coat)  # in RGB format
img_coat1 = ImageOps.grayscale(img_coat)
img_coat2 = img_coat1.resize((28,28))
#img_coat.show()
img_coat2.show()
img_coat3 = np.expand_dims(img_coat2, axis=0)

predCoat2 = loaded_model.predict(img_coat3) 
#img_coat4 = np.expand_dims(x_test[1], axis=0)
#predCoat2 = loaded_model.predict(img_coat4)
print("Probabilities of each category for imag_purse3: \n")
print(predCoat2[0], '\n')
print('Shape of predCoat2[0]: ', predCoat2[0].shape, '\n')
print("Predicted category for x_test[0] image: \n")
print(class_names[np.argmax(predCoat2[0])]) # np.argmax(a_list or an_array) returns 
# the index of the maximum item in a list or the indices of the maximum elements
# in an array along one axis
print("Display img_coat3[0]: \n")
plt.figure()
plt.imshow(img_coat3[0])
plt.colorbar()
plt.show()

