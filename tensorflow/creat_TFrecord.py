import os
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_and_decode(filename):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [28, 28, 1])
    #img = tf.reshape(img, [784])
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #Convert `image` from [0, 255] -> [-0.5, 0.5] floats 归一化，会减少损失函数的震荡，有助于减小损失函数提高精度
    label = tf.cast(features['label'], tf.int32)

    return img, label

if __name__ == '__main__':
    create_record('data/train')
    create_record('data/test')
    img, label = read_and_decode("train.tfrecords")
    BATCH_SIZE=10
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=BATCH_SIZE, capacity=2000,
                                                    min_after_dequeue=1000)
   
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #队列监控
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(2):
            img, label= sess.run([img_batch, label_batch])
            for j in range(BATCH_SIZE):
            
                print(label[j])
             
                cv2.imshow("s",img[j,:,:,:])

                cv2.waitKey(0)
           
     
      
      
       
        
        
        coord.request_stop()
        coord.join(threads) #把开启的线程加入主线程，等待threads结束
        
        
        
        
        
        
        
        
        
        