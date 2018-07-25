import glob
import os
import cv2
import h5py
import numpy as np
import tensorflow as tf
from PIL import Image



def create_folder(class_num):
    for i in range(class_num):
        if not os.path.exists('train/'+str(i)):
            os.makedirs('train/'+str(i))
    print('%d 类文件夹生成成功' % (class_num))


def create_mydataset():
    path = 'train'
    fw = open('my_dataset.txt', 'w')
    image_files = os.listdir(path)
    image_files.sort(key = int)
    for i in range(len(image_files)):
        dog_categories = os.listdir(path+'/'+image_files[i])
        for each_image in dog_categories:
            fw.write(path+'/'+image_files[i]+'/'+ each_image + ' %d\n'% i)
    print('生成my_dataset.txt文件成功\n')
    fw.close()

    
def rename_img(path):
    i = 9000
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    for files in filelist:  # 遍历所有文件
        i = i-1
        Olddir = os.path.join(path, files)  # 原来的文件路径                
        if os.path.isdir(Olddir):  # 如果是文件夹则跳过
                continue
        filename = os.path.splitext(files)[0]  # 文件名
        filetype = os.path.splitext(files)[1]  # 文件扩展名
        Newdir = os.path.join(path, str(i)+filetype)  # 新的文件路径
        os.rename(Olddir, Newdir)  # 重命名
    print('over')
    

def rename_folder(path):
    filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
    filelist.sort(key = int)
    i = 0
    for files in filelist:  # 遍历所有文件
        Olddir = os.path.join(path, files)  # 原来的文件路径                
        if os.path.isdir(Olddir):
            if str(files) != i:
                Newdir = os.path.join(path, str(i))
                os.rename(Olddir, Newdir)  # 重命名
        i += 1
    print('over')


def mnist_to_img():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist_data = input_data.read_data_sets('MNIST_data/', one_hot=False)
    images = mnist_data.train.images
    labels = mnist_data.train.labels
    num_examples = mnist_data.train.num_examples
    for i, (arr, label) in enumerate(zip(images, labels)):
        if not os.path.exists('train/'+str(label)):
            os.makedirs('train/'+str(label))
        # print(i, label)
        # 直接保存 arr，是黑底图片，1.0 - arr 是白底图片
        matrix = (np.reshape(1.0 - arr, (28, 28)) * 255).astype(np.uint8)
        img = Image.fromarray(matrix, 'L')
        # 存储图片时，label_index的格式，方便在制作数据集时，从文件名即可知道label
        img.save('train/'+str(label)+"/{}_{}.png".format(label, i))
    print('over')


def img_generator(img_path):
    from keras.preprocessing.image import ImageDataGenerator, load_img
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (50, 50))
    img = np.array(img) 
    img = img.reshape(-1, 50, 50, 1)
    i = 0
    for batch in datagen.flow(img, batch_size=1,
                            save_to_dir='preview', save_prefix='0', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely


'''
csv
'''
def create_csv():
    import csv
    imgarray = []
    list = os.listdir('train')  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join('train', list[i])
        imgs = glob.glob(path+'\*')
        for img in imgs:   
            img = Image.open(img).convert('L')
            img = img.resize((50, 50))
            img_ndarray = np.asarray(img, dtype='float64') / 255.0   # 归一化
            img_ndarray = np.ndarray.flatten(img_ndarray)      # 压平为一维向量  一维数组
            img_ndarray1 = img_ndarray.tolist()                 # 将array格式转为list格式
            img_ndarray1.extend([i])                    # 末尾添加
            imgarray.append(img_ndarray1)                # 只有转化为list格式才能使用append方法
    with open('train.csv', 'w') as myfile:
        mywriter = csv.writer(myfile)
        mywriter.writerows(imgarray)          # list格式才可以使用mywriter写入  array和matrix格式都不可以
        print('csv创建成功')


'''
npy
'''
def create_npy(w, h, class_num):
    X, Y = [], []
    with open('my_dataset.txt','r') as f:
        for line in f:  
            img_path, img_label = line.strip().split(' ')
            img = Image.open(img_path).convert('L')
            img = img.resize((w, h))
            img = 1 - np.reshape(img, w*h) / 255.0   # 图片像素值映射到0 - 1之间
            X.append(img)
            label_one_hot = [0 if i != int(img_label) else 1 for i in range(class_num)]
            Y.append(label_one_hot)
    np.save('X.npy', np.array(X))
    np.save('Y.npy', np.array(Y)) 
    print('y[0]:' + str(y[0]))
    cv2.imshow('x[0]',x[0])
    cv2.waitKey(0)
    print('npy创建成功')
   

def read_npy(w, h, channel=1, shuffle=True):
    x, y = np.load('X.npy'), np.load('Y.npy')
    x = x.reshape(-1, w, h, channel)
    print('x.shape:  ' + str(x.shape))
    print('y.shape:  ' + str(y.shape))
    if shuffle:
        print('shuffle前y[0]:  ' + str(y[0]))
        index=np.arange(len(y))
        np.random.shuffle(index)
        x=x[index] #X_train是训练集，y_train是训练标签
        y=y[index]
        print('shuffle后y[0]:  ' + str(y[0]))
    return x, y



'''
h5py
'''
def create_h5py(w, h, class_num, shuffle=True):  
    h5f = h5py.File('train.h5', 'w')
    with open('my_dataset.txt', 'r') as f:
        img_num = len(f.readlines())
    x = h5f.create_dataset("my_datas", (img_num, w, h, 1), dtype='float32')
    y = h5f.create_dataset("my_labels", (img_num, class_num), dtype='int32')
    temp_x = np.empty((img_num, w, h, 1), dtype="float32")
    temp_y = np.empty((img_num, class_num), dtype="int32")
    i = 0
    with open('my_dataset.txt','r') as f:
        for line in f:  
            img_path, img_label = line.strip().split(' ')
            one_hot_labels = [0 if i != int(img_label) else 1 for i in range(class_num)]
            img = Image.open(img_path).convert('L')
            img = img.resize((w, h))
            img = 1-np.array(img) / 255.0  # 图片像素值映射到 0 - 1之间
            temp_x[i, :, :, :] = np.asarray(img, dtype="float32").reshape(w, h, 1)
            temp_y[i, :] = one_hot_labels
            i += 1
    if shuffle:
        print('shuffle前y[0]:  ' + str(temp_y[0]))
        index=np.arange(len(temp_y))
        np.random.shuffle(index)
        temp_x=temp_x[index] #X_train是训练集，y_train是训练标签
        temp_y=temp_y[index]
        print('shuffle后y[0]:  ' + str(temp_y[0]))
    x[:, :, :, :] = temp_x
    y[:, :] = temp_y
    print('y.shape:  ' + str(y.shape))
    print('x.shape:  ' + str(x.shape))    
    print('y[0]:' + str(y[0]))
    cv2.imshow('x[0]',x[0])
    cv2.waitKey(0)       
    h5f.close()
    print('h5py创建成功')


def create_h5py_batch(batchsize, w, h, class_num):
    h5f = h5py.File('train.h5', 'w')
    dataset = h5f.create_dataset("my_datas", (batchsize, w, h, 1), maxshape=(None, w, h, 1), dtype='float32')
    labels = h5f.create_dataset("my_labels", (batchsize, class_num), maxshape=(None, class_num), dtype='int32')
    temp_data = np.empty((batchsize, w, h, 1), dtype="float32")
    temp_label = np.empty((batchsize, class_num), dtype="int32")
    i = 0
    times = 0
    with open('my_dataset.txt', 'r') as f:
        for line in f:  
            img_path, img_label = line.strip().split(' ')
            one_hot_labels = [0 if i != int(img_label) else 1 for i in range(class_num)]
            img = Image.open(img_path).convert('L')  # 灰度化
            img = img.resize((w, h))
            img = 1-np.array(img) / 255.0  # 图片像素值映射到 0 - 1之间 float类型
            temp_data[i, :, :, :] = np.asarray(img, dtype="float32").reshape(w, h, 1)
            temp_label[i, :] = one_hot_labels
            # print(i)
            i += 1   
            if i == batchsize:
                dataset.resize([times*batchsize + batchsize, w, h, 1])
                dataset[times*batchsize:times*batchsize+batchsize, :, :, :] = temp_data
                labels.resize([times*batchsize + batchsize, class_num])
                labels[times*batchsize:times*batchsize+batchsize, :] = temp_label
                # print(labels.shape)
                # print(dataset.shape)
                times += 1
                i = 0        
    h5f.close()
    print('h5py创建成功')     


def read_h5py():
    h5f = h5py.File('train.h5', 'r')
    x = h5f['my_datas']
    y = h5f['my_labels']
    print('x.shape:  ' + str(x.shape))
    print('y.shape:  ' + str(y.shape))
    return x, y



'''
TFrecords
'''
def create_TFrecords(path, w, h,  classes_num):
    num_classes = [i for i in range(classes_num)]
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(path+".tfrecords")
    for index, name in enumerate(num_classes):
        class_path = cwd + "\\"+path+"\\" + str(name) + "\\"
        for img_name in os.listdir(class_path):
            img = Image.open(class_path + img_name).convert('L')  # 转化为灰度
            img = img.resize((w, h))
            img_raw = img.tobytes()  # 将图片转化为原生bytes
            example = tf.train.Example(features=tf.train.Features(feature={
                "my_labels": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'my_datas': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
    writer.close()
    print('TFrecord创建成功')


def read_TFrecords(filename):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'my_labels': tf.FixedLenFeature([], tf.int64),
                                           'my_datas': tf.FixedLenFeature([], tf.string), })
    img = tf.decode_raw(features['my_datas'], tf.uint8)
    img = tf.reshape(img, [25, 25, 1])
    img = 1 - tf.cast(img, tf.float32) / 255.0
    label = tf.cast(features['my_labels'], tf.int32)
    return img, label


def get_TFbatch(imgs, labels, n_classes, batch_size):
    imgs_batch, labels_batch = tf.train.shuffle_batch([imgs, labels],batch_size=batch_size,capacity=2000,min_after_dequeue=1000)
    labels_batch = tf.one_hot(labels_batch, n_classes)
    labels_batch = tf.cast(labels_batch, dtype=tf.int64)
    labels_batch = tf.reshape(labels_batch, [batch_size, n_classes])
    return imgs_batch, labels_batch


def test_TFrecords(n_classes=10, batch_size=64):
    img, label=read_TFrecords('train.tfrecords')
    imgs_batch, labels_batch=get_TFbatch(img, label, n_classes, batch_size)
    with tf.Session() as sess:
        coord=tf.train.Coordinator()#创建一个协调器，管理线程  若数据非TFrecord 则可不需要这两句
        threads= tf.train.start_queue_runners(sess=sess, coord=coord)#启动QueueRunner, 此时文件名队列已经进队 TFrecord数据必须开启队列，否则tensorflow将一直挂起
        img, label=sess.run([imgs_batch, labels_batch])
        print('y.shape:  ' + str(label.shape)+'  x.shape:  ' + str(img.shape))
        print('y[0]:' + str(label[0]))
        cv2.imshow('x[0]',img[0])
        cv2.waitKey(0) 
        coord.request_stop()
        coord.join(threads) #把开启的线程加入主线程，等待threads结束






