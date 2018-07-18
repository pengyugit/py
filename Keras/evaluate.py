import numpy as np
from keras.models import load_model
from PIL import Image  

model = load_model('my_model.h5')

img = Image.open('1.png').convert('L')
img = img.resize((28, 28))
img = 1-np.array(img) / 255.0

# X=np.array(X, 'float32') # 归一化数据
# testX=np.array(testX, 'uint8')
img= img.reshape([-1, 28, 28, 1])

output1=model.predict(img)[0]
output1 = output1.tolist()


print(output1.index(max(output1)))
print(output1)


