import tensorflow as tf
import model
import cv2
import numpy as np

x = tf.placeholder("float", [None, 28*28])
sess = tf.Session()
with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "model/convolutional.ckpt")


img = cv2.imread('0.png',0)
img=cv2.resize(img,(28,28))

input = ((255 - np.array(img, dtype=np.uint8)) / 255.0)
result=sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()
print(result)


img = cv2.imread('1.png',0)
img=cv2.resize(img,(28,28))
input = ((255 - np.array(img, dtype=np.uint8)) / 255.0).reshape(1, 28, 28 ,1)
result=sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()
print(result)

img = cv2.imread('2.png',0)
img=cv2.resize(img,(28,28))
input = ((255 - np.array(img, dtype=np.uint8)) / 255.0).reshape(1, 28, 28 ,1)
result=sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()
print(result)

img = cv2.imread('3.png',0)
img=cv2.resize(img,(28,28))
input = ((255 - np.array(img, dtype=np.uint8)) / 255.0).reshape(1, 28, 28 ,1)
result=sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()
print(result)

img2 = cv2.imread('4.png',0)
img2=cv2.resize(img2,(28,28))
input2 = ((255 - np.array(img2, dtype=np.uint8)) / 255.0).reshape(1, 28, 28 ,1)
result2=sess.run(y2, feed_dict={x: input2, keep_prob: 1.0}).flatten().tolist()
print(result2)
    
