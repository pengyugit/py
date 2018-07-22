import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import load_csv
import numpy as np
import cv2


##convnet = input_data(shape=[None, 28, 28, 1], name='input')
##convnet = conv_2d(convnet, 32, 5, activation='relu')
##convnet = max_pool_2d(convnet, 5)
##convnet = conv_2d(convnet, 64, 5, activation='relu')
##convnet = max_pool_2d(convnet, 5)
##convnet = conv_2d(convnet, 128, 5, activation='relu')
##convnet = max_pool_2d(convnet, 5)
##convnet = conv_2d(convnet, 64, 5, activation='relu')
##convnet = max_pool_2d(convnet, 5)
##convnet = conv_2d(convnet, 32, 5, activation='relu')
##convnet = max_pool_2d(convnet, 5)
##convnet = fully_connected(convnet, 1024, activation='relu')
##convnet = dropout(convnet, 0.8)
##convnet = fully_connected(convnet, 10, activation='softmax')
##model2 = tflearn.DNN(convnet)


convnet = input_data(shape=[None, 28, 28, 1], name='input')
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = dropout(convnet, 0.25)
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = dropout(convnet, 0.25)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.5)
convnet = fully_connected(convnet, 81, activation='softmax')

model2 = tflearn.DNN(convnet)

model2.load('my_model.tflearn')





#img = cv2.imread('train/8/100.jpg',0)
img = cv2.imread('2.jpg',0)
img=cv2.resize(img,(28,28))
img=np.array(img, dtype=np.uint8).reshape(-1, 28, 28, 1)
img = 1 - img/ 255.0  
output1=model2.predict(img)[0]
output1 = output1.tolist()


print(output1.index(max(output1)))
print(max(output1))








