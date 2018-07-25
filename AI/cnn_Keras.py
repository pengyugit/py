import keras
from keras.utils.io_utils import HDF5Matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
import cv2
from DataSet import *

class_num = 10
img_w = 25
img_h = 25
x_train, y_train =read_h5py()


print(y_train[0])
cv2.imshow('ss',x_train[0] )
cv2.waitKey(0)


model = Sequential()
model.add(Conv2D(32, (3, 3), strides=1, activation='relu', input_shape=(img_w, img_h, 1)))
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
model.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=1, validation_split=0.1, shuffle="batch")
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
