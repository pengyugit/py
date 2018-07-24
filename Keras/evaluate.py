import numpy as np
from keras.models import load_model
import cv2

model = load_model('my_model.h5')
# img = cv2.imread('2.jpg', 0)
# img = cv2.resize(img, (28, 28))
# img = 1 - np.array(img) / 255.0
# ##cv2.imshow('s', img)
# ##cv2.waitKey(0)
# # X=np.array(X, 'float32') # 归一化数据
# # testX=np.array(testX, 'uint8')
# img= img.reshape([-1, 28, 28, 1])

# output1=model.predict(img)[0]
# output1 = output1.tolist()


# print(output1.index(max(output1)))
# print(max(output1))
# #print(output1)


# x_test, y_test = np.load('X.npy'), np.load('Y.npy')
# x_test = x_test.reshape(-1, 50, 50, 1)
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])


# from keras.utils import plot_model
# plot_model(model, to_file='model.png')