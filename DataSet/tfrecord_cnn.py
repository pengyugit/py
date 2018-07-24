#coding=utf-8
import os
import model
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def read_and_decode(filename):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
   # img = tf.reshape(img, [28, 28, 1])
    img = tf.reshape(img, [784])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #Convert `image` from [0, 255] -> [-0.5, 0.5] floats
    label = tf.cast(features['label'], tf.int32)
    return img, label


def get_batch(imgs, labels, n_classes, batch_size):
    imgs_batch, labels_batch = tf.train.shuffle_batch([imgs, labels],batch_size=batch_size,capacity=2000,min_after_dequeue=1000)
    labels_batch = tf.one_hot(labels_batch, n_classes)
    labels_batch = tf.cast(labels_batch, dtype=tf.int64)
    labels_batch = tf.reshape(labels_batch, [batch_size, n_classes])
    return imgs_batch, labels_batch



# model   
with tf.variable_scope("convolutional"):
    x = tf.placeholder(tf.float32, [None, 28*28*1])# 声明一个占位符，None表示输入图片的数量不定，28*28图片分辨率
    # 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
    keep_prob = tf.placeholder(tf.float32)  	
    y, variables = model.convolutional(x, keep_prob)

y_ = tf.placeholder(tf.float32, [None, 3])# 类别是0-9总共10个类别，对应输出分类结果

#global_step = tf.Variable(0, trainable=False)
#learning_rate = tf.train.exponential_decay(0.03,current_epoch,decay_steps=num_epochs,decay_rate=0.03)
# 定义loss(最小误差概率)，选定优化优化loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y))
cross_entropy = -tf.reduce_sum(y_ *tf.log(y)) # 定义交叉熵为loss函数 
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cross_entropy)# 调用优化器优化，通过喂数据争取cross_entropy最小化 
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


train_imgs, train_labels = read_and_decode("data/train.tfrecords")
img_test, label_test = read_and_decode("data/test.tfrecords")
train_imgs_batch, train_label_batch = get_batch(train_imgs, train_labels, n_classes=3, batch_size=50)
val_imgs_batch, val_label_batch = get_batch(img_test, label_test ,  n_classes=3, batch_size=50)



   
saver = tf.train.Saver(variables)
summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('log/log_train_dir', sess.graph)
    val_writer = tf.summary.FileWriter('log/logs_val_dir', sess.graph)
    sess.run(tf.global_variables_initializer())
    coord=tf.train.Coordinator()#创建一个协调器，管理线程
    threads= tf.train.start_queue_runners(sess=sess, coord=coord)#启动QueueRunner, 此时文件名队列已经进队 训练步骤之前，需要调用tf.train.start_queue_runners函数，否则tensorflow将一直挂起

    for step in range(600):
        if coord.should_stop():
            break
        train_batch, train_labels = sess.run([train_imgs_batch, train_label_batch])
        val_images, val_labels = sess.run([val_imgs_batch, val_label_batch])
        
        _ , train_loss, train_acc = sess.run([train_op, loss, accuracy], feed_dict={x: train_batch, y_: train_labels, keep_prob: 0.5})
        val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})
        if step % 10 == 0:
            print('Step %d, loss %f, acc %.2f%% --- * val_loss %f, val_acc %.2f%%' % (step, train_loss ,train_acc * 100.0, val_loss ,val_acc * 100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
            val_writer.add_summary(summary_str, step)

    checkpoint_path = os.path.join( 'model', 'convolutional.ckpt')
   # saver.save(sess, checkpoint_path, global_step=step)
    saver.save(sess, checkpoint_path, write_meta_graph=False, write_state=False)
    
    
    coord.request_stop()
    coord.join(threads) #把开启的线程加入主线程，等待threads结束
    print('all threads are stopped!')

    