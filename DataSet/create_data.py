from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import cv2





datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
 
#img = load_img('train/0/0.jpg')  # this is a PIL image
img = cv2.imread('train/0/0.jpg', 0)
img = cv2.resize(img, (50, 50))
img = np.array(img) 
img = img.reshape(-1, 50, 50, 1)
# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
 
# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(img, batch_size=1,
                          save_to_dir='preview', save_prefix='0', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
