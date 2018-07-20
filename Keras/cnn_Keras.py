import numpy as np
import keras
from keras.utils.io_utils import HDF5Matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.datasets import cifar100
from keras.models import load_model
import cv2

#(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')



def load_data(path, start_ix, n_samples):
    X = HDF5Matrix(path, 'my_datas', start_ix, start_ix + n_samples)
    y = HDF5Matrix(path, 'my_labels', start_ix, start_ix + n_samples)
    return (X,y)
#x_train, y_train = load_data(r'dataset/train.h5', 0, 5)

x_train, y_train = np.load('X.npy'), np.load('Y.npy')
x_train = x_train.reshape(-1, 28, 28, 1)

# x_train = HDF5Matrix(r'train.h5', 'my_datas')
# y_train = HDF5Matrix(r'train.h5', 'my_labels')
class_num=81

# print(y_train.shape)
# print(x_train.shape)
# print(y_train[20008])
# cv2.imshow('ss',x_train[20008] )
# cv2.waitKey(0)


model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), strides=1, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(class_num, activation='softmax'))

Adam=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy', optimizer=Adam ,metrics=['accuracy'])




print('shuffle前:  ' + str(y_train[0]))
index=np.arange(len(y_train))
np.random.shuffle(index)
x_train=x_train[index] #X_train是训练集，y_train是训练标签
y_train=y_train[index]
print('shuffle后:  ' + str(y_train[0]))


model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, validation_split=0.2, shuffle="batch")
model.save('my_model.h5')

# score = model.evaluate(x_test, y_test, batch_size=32)
# print(score)


# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 128, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = fully_connected(convnet, 1024, activation='relu')
# convnet = dropout(convnet, 0.8)
