import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import load_csv
import numpy as np
#import cv2

# Load path/class_id image file:
# /path/to/img1 class_id
# /path/to/img2 class_id
# /path/to/img3 class_id
dataset_file = 'my_dataset.txt'


import h5py
h5f = h5py.File('train.h5', 'r')
X = h5f['my_datas']
Y = h5f['my_labels']





# from tflearn.data_utils import image_preloader
# X, Y = image_preloader(dataset_file, image_shape=(50, 50),   mode='file', categorical_labels=True,   normalize=False ,grayscale=True)

print(X[0].shape)
print(Y[4])
print('over')








X=np.array(X, 'uint8')
# testX=np.array(testX, 'uint8')
X = X.reshape([-1, 28, 28, 1])
# testX = testX.reshape([-1, 50, 50, 1])
# cv2.imshow("s",X[0,:,:,:])
# cv2.waitKey(0)



# convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# convnet = conv_2d(convnet, 32, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = conv_2d(convnet, 64, 5, activation='relu')
# convnet = max_pool_2d(convnet, 5)
# convnet = fully_connected(convnet, 1024, activation='relu')
# convnet = dropout(convnet, 0.8)
# convnet = fully_connected(convnet, 2, activation='softmax')
# convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
# model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
# model.fit({'input': X_train}, {'targets': y_train}, n_epoch=10, 
          # validation_set=({'input': X_test}, {'targets': y_test}), 
          # snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
convnet = input_data(shape=[None, 28, 28, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 81, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_verbose=3, tensorboard_dir='logs')


# network = input_data(shape=[None, 50, 50, 1], name='input')
# network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
# network = max_pool_2d(network, 2)
# network = local_response_normalization(network)
# network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
# network = max_pool_2d(network, 2)
# network = local_response_normalization(network)
# network = fully_connected(network, 128, activation='tanh')
# network = dropout(network, 0.8)
# network = fully_connected(network, 256, activation='tanh')
# network = dropout(network, 0.8)
# network = fully_connected(network, 2, activation='softmax')
# network = regression(network, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy', name='target')



#model = tflearn.DNN(network, tensorboard_verbose=3)   validation_set=(testX, testY),
model.fit( X,  Y, n_epoch=2, show_metric=True, batch_size = 50, run_id = 'py')
           
model.save('my_model.tflearn')