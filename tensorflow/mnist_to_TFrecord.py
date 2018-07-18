# -*- coding: utf-8 -*- 
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 生成整数的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets('MNIST_data/', dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels

num_examples = mnist.train.num_examples


filename = 'output1.tfrecords'
writer = tf.python_io.TFRecordWriter(filename)

#将每张图片都转为一个Example
for i in range(num_examples):
    image_raw = images[i].tostring()  # 将图像转为字符串
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'label': _int64_feature(np.argmax(labels[i])),
            'img_raw': _bytes_feature(image_raw)
        }))
    writer.write(example.SerializeToString())  # 将Example写入TFRecord文件

print('data processing success')
writer.close()