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

print('shuffle前:  ' + str(y_train[0]))
index=np.arange(len(y_train))
np.random.shuffle(index)
x_train=x_train[index] #X_train是训练集，y_train是训练标签
y_train=y_train[index]
print('shuffle后:  ' + str(y_train[0]))

# x_train = HDF5Matrix(r'train.h5', 'my_datas')
# y_train = HDF5Matrix(r'train.h5', 'my_labels')
class_num=81

# print(y_train.shape)
# print(x_train.shape)
# print(y_train[20008])
# cv2.imshow('ss',x_train[20008] )
# cv2.waitKey(0)


model = Sequential()
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

model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=1, validation_split=0.5, shuffle="batch")
model.save('my_model.h5')

# score = model.evaluate(x_test, y_test, batch_size=32)
# print(score)




# lenet = Sequential()
# lenet.add(Conv2D(6, kernel_size=3, strides=1, padding='same', input_shape=(28, 28, 1)))
# lenet.add(MaxPooling2D(pool_size=2, strides=2))
# lenet.add(Conv2D(16, kernel_size=5, strides=1, padding='valid'))
# lenet.add(MaxPooling2D(pool_size=2, strides=2))
# lenet.add(Flatten())#多维向量压成一维
# lenet.add(Dense(120))
# lenet.add(Dense(84))
# lenet.add(Dense(81, activation='softmax'))

# lenet.summary()
# lenet.compile('sgd', loss='categorical_crossentropy', metrics=['accuracy'])
# lenet.fit(x_train, y_train, batch_size=64, epochs=50, validation_split=0.5, shuffle="batch")
