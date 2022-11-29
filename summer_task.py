from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
import pandas._libs.tslibs.base
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import unit10.utils as u10
from DL4 import *
import tensorflow as tf
from keras.datasets import cifar10

(X, Y), (test_X, test_Y) = cifar10.load_data()

Y = Y.reshape(-1).T
test_Y = test_Y.reshape(-1).T

#flatten
train_x_flatten = X.reshape(X.shape[0], -1).T
test_x_flatten = test_X.reshape(test_X.shape[0], -1).T

#normalize
X = train_x_flatten/255.0 - 0.5
test_X = test_x_flatten/255.0 - 0.5

#transpose
X = X.T

print(X.shape)
print(test_X.shape)

Y = np.array(Y)
X = np.array(X)
classes = 10
examples = Y.shape[0]
Y = Y.reshape(1, examples)

examples_test = test_Y.shape[0]
test_Y = test_Y.reshape(1, examples_test)

Y_new = np.eye(classes)[Y.astype('int32')]
Y_new = Y_new.T.reshape(classes, examples)

Y_new_test = np.eye(classes)[test_Y.astype('int32')]
Y_new_test = Y_new_test.T.reshape(classes, examples_test)

train_X = X.T
train_Y = np.array(Y_new[:,:50000])
test_Y = np.array(Y_new_test[:,:10000])


np.random.seed(1)
layer = DLLayer("layer 1", 64,(3072,), activation = "relu", W_initialization = "He", learning_rate = 1)
hidden_layer1 = DLLayer("hidden 1", 64,(64,), activation = "relu", W_initialization = "He", learning_rate = 1)
hidden_layer2 = DLLayer("hidden 2", 64,(64,), activation = "relu", W_initialization = "He", learning_rate = 1)
hidden_layer3 = DLLayer("hidden 3", 32,(64,), activation = "sigmoid", W_initialization = "He", learning_rate = 1)
output_layer = DLLayer("output", 10,(32,), activation = "softmax", W_initialization = "He", learning_rate = 1)
model = DLModel()
model.add(layer)
model.add(hidden_layer1)
model.add(hidden_layer2)
model.add(hidden_layer3)
model.add(output_layer)
model.compile("categorical_cross_entropy")
costs = model.train(train_X, train_Y, 100)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(1))
plt.show()

def predict_softmax(X, Y, model):
    AL = model.predict(X)
    predictions = np.argmax(AL, axis=0)
    labels = np.argmax(Y, axis=0)
    return predictions, labels

train_predictions, train_labels = predict_softmax(train_X, train_Y, model)
test_predictions, test_labels = predict_softmax(test_X, test_Y, model)
print('Deep train accuracy')
pred_train = confusion_matrix(train_predictions, train_labels)
print(pred_train)
print(test_X.shape)
print('Deep test accuracy')
pred_test = confusion_matrix(test_predictions, test_labels)
print(pred_test)



print("train accuracy:" + str(np.mean(train_predictions == train_labels)) + "%")
print("test accuracy:" + str(np.mean(test_predictions == test_labels)) + "%")

#print("train accuracy:" + str(np.mean(model.predict(train_X) == train_Y) * 100) + "%")
#print("test accuracy:" + str(np.mean(model.predict(test_X) == test_Y) * 100) + "%")