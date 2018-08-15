import keras
from keras.utils.io_utils import HDF5Matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import load_model
import cv2
from DataSet import *


def model1(class_num, img_w, img_h, channel=1):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), strides=1, activation='relu', input_shape=(img_w, img_h, channel)))
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
    tb = keras.callbacks.TensorBoard(log_dir='./logs',  # log 目录
                    histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                    batch_size=32,     # 用多大量的数据计算直方图
                    write_graph=True,  # 是否存储网络结构图
                    write_grads=False, # 是否可视化梯度直方图
                    write_images=False,# 是否可视化参数
                    embeddings_freq=0, 
                    embeddings_layer_names=None, 
                    embeddings_metadata=None)
    callbacks = [tb]
    model.fit(x_train, y_train, batch_size=64, epochs=1000, verbose=1, validation_split=0.1, shuffle="batch",callbacks=callbacks)
    model.save('my_model.h5')




def model_lenet(class_num, img_w, img_h, channel=1):
    lenet = Sequential()
    lenet.add(Conv2D(6, kernel_size=3, strides=1, padding='same', input_shape=(28, 28, 1)))
    lenet.add(MaxPooling2D(pool_size=2, strides=2))
    lenet.add(Conv2D(16, kernel_size=5, strides=1, padding='valid'))
    lenet.add(MaxPooling2D(pool_size=2, strides=2))
    lenet.add(Flatten())#多维向量压成一维
    lenet.add(Dense(120))
    lenet.add(Dense(84))
    lenet.add(Dense(81, activation='softmax'))
    lenet.summary()
    lenet.compile('sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    lenet.fit(x_train, y_train, batch_size=64, epochs=50, validation_split=0.5, shuffle="batch")



def evaluate():
    model = load_model('my_model.h5')
    img = cv2.imread('2.jpg', 0)
    img = cv2.resize(img, (28, 28))
    img = 1 - np.array(img) / 255.0
    cv2.imshow('s', img)
    cv2.waitKey(0)
    img= img.reshape([-1, 28, 28, 1])
    output1=model.predict(img)[0]
    output1 = output1.tolist()
    print(output1.index(max(output1)))
    print(max(output1))
    print(output1)

    x_test, y_test = np.load('X.npy'), np.load('Y.npy')
    x_test = x_test.reshape(-1, 50, 50, 1)
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    from keras.utils import plot_model
    plot_model(model, to_file='model.png')




if __name__ == "__main__":
    x_train, y_train =read_h5py()
    print(y_train[0])
    cv2.imshow('ss',x_train[0] )
    cv2.waitKey(0)
    model1(class_num=10, img_w=25, img_h=25)
