import numpy as np
from keras.models import load_model
from PIL import Image  
import cv2
model = load_model('my_model.h5')

img = Image.open('2.png').convert('L')
img = img.resize((28, 28))
img = 1 - np.array(img) / 255.0
##cv2.imshow('s', img)
##cv2.waitKey(0)
# X=np.array(X, 'float32') # 归一化数据
# testX=np.array(testX, 'uint8')
img= img.reshape([-1, 28, 28, 1])

output1=model.predict(img)[0]
output1 = output1.tolist()


print(output1.index(max(output1)))
print(output1)


