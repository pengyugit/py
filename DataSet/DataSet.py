import csv
import glob
import os

import cv2
import h5py
import numpy as np
import tensorflow as tf
from PIL import Image


def create_mydataset():
    path = 'train'
    fw = open('my_dataset.txt', 'w')
    image_files = os.listdir(path)
    for i in range(len(image_files)):
        dog_categories = os.listdir(path+'/'+image_files[i])
        for each_image in dog_categories:
            fw.write(path+'/'+image_files[i]+'/' + each_image + ' %d\n'% i)
    print('生成my_dataset.txt文件成功\n')
    fw.close()

    
def create_folder(class_num):
    for i in range(class_num):
        if not os.path.exists(str(i)):
            os.makedirs(str(i))
    print('%d 类文件夹生成成功' % (class_num))

    
def rename(path):
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
    

# class DataSet:
#     def __init__(self):
#         with h5py.File('./data_set/data.h5', 'r') as f:
#             x, y = f['x_data'].value, f['y_data'].value

#         self.train_x, self.test_x, self.train_y, self.test_y = \
#             train_test_split(x, y, test_size=0.2, random_state=0)

#         self.train_size = len(self.train_x)

#     def get_train_batch(self, batch_size=64):
#         # 随机获取batch_size个训练数据
#         choice = np.random.randint(self.train_size, size=batch_size)
#         batch_x = self.train_x[choice, :]
#         batch_y = self.train_y[choice, :]

#         return batch_x, batch_y

#     def get_test_set(self):
#         return self.test_x, self.test_y


def create_csv():
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

 
def create_npy(img_width, img_heigh, class_num):
    X, Y = [], []
    with open('my_dataset.txt','r') as f:
        for line in f:  
            img_path, img_label = line.strip().split(' ')
            img = Image.open(img_path).convert('L')
            img = img.resize((img_width, img_heigh))
            img = 1 - np.reshape(img, img_width*img_heigh) / 255.0   # 图片像素值映射到0 - 1之间
            X.append(img)
            label_one_hot = [0 if i != int(img_label) else 1 for i in range(class_num)]
            Y.append(label_one_hot)
    np.save('X.npy', np.array(X))
    np.save('Y.npy', np.array(Y)) 
    print('npy创建成功')
   

def create_h5py(img_width, img_heigh, class_num):  
    h5f = h5py.File('train.h5', 'w')
    with open('my_dataset.txt', 'r') as f:
        img_num = len(f.readlines())
    dataset = h5f.create_dataset("my_datas", (img_num, img_width, img_heigh, 1), dtype='float32')
    labels = h5f.create_dataset("my_labels", (img_num, class_num), dtype='int32')
    temp_data = np.empty((img_num, img_width, img_heigh, 1), dtype="float32")
    temp_label = np.empty((img_num, class_num), dtype="int32")
    i = 0
    with open('my_dataset.txt','r') as f:
        for line in f:  
            img_path, img_label = line.strip().split(' ')
            one_hot_labels = [0 if i != int(img_label) else 1 for i in range(class_num)]
            img = Image.open(img_path).convert('L')
            img = img.resize((img_width, img_heigh))
            img = 1-np.array(img) / 255.0  # 图片像素值映射到 0 - 1之间
            temp_data[i, :, :, :] = np.asarray(img, dtype="float32").reshape(img_width, img_heigh, 1)
            temp_label[i, :] = one_hot_labels
            i += 1

    dataset[:, :, :, :] = temp_data
    labels[:, :] = temp_label
          
    print(labels.shape)
    print(dataset.shape)               
    h5f.close()
    print('h5py创建成功')


def create_h5py_batch(batchsize, img_width, img_heigh, class_num):
    h5f = h5py.File('train.h5', 'w')
    dataset = h5f.create_dataset("my_datas", (batchsize, img_width, img_heigh, 1), maxshape=(None, img_width, img_heigh, 1), dtype='float32')
    labels = h5f.create_dataset("my_labels", (batchsize, class_num), maxshape=(None, class_num), dtype='int32')
    temp_data = np.empty((batchsize, img_width, img_heigh, 1), dtype="float32")
    temp_label = np.empty((batchsize, class_num), dtype="int32")
    i = 0
    times = 0
    with open('my_dataset.txt', 'r') as f:
        for line in f:  
            img_path, img_label = line.strip().split(' ')
            one_hot_labels = [0 if i != int(img_label) else 1 for i in range(class_num)]
            img = Image.open(img_path).convert('L')  # 灰度化
            img = img.resize((img_width, img_heigh))
            img = 1-np.array(img) / 255.0  # 图片像素值映射到 0 - 1之间 float类型
            temp_data[i, :, :, :] = np.asarray(img, dtype="float32").reshape(img_width, img_heigh, 1)
            temp_label[i, :] = one_hot_labels
            # print(i)
            i += 1   
            if i == batchsize:
                dataset.resize([times*batchsize + batchsize, img_width, img_heigh, 1])
                dataset[times*batchsize:times*batchsize+batchsize, :, :, :] = temp_data
                labels.resize([times*batchsize + batchsize, class_num])
                labels[times*batchsize:times*batchsize+batchsize, :] = temp_label
                # print(labels.shape)
                # print(dataset.shape)
                times += 1
                i = 0        
    h5f.close()
    print('h5py创建成功')
               

def create_TFrecords(path, img_width, img_heigh,  classes_num):
    num_classes = [i for i in range(classes_num)]
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(path+".tfrecords")
    for index, name in enumerate(num_classes):
        class_path = cwd + "\\"+path+"\\" + str(name) + "\\"
        for img_name in os.listdir(class_path):
            img = Image.open(class_path + img_name).convert('L')  # 转化为灰度
            img = img.resize((img_width, img_heigh))
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
    img = tf.reshape(img, [28, 28, 1])
    img = 1 - tf.cast(img, tf.float32) / 255.0
    label = tf.cast(features['my_labels'], tf.int32)
    return img, label


def get_batch_TFrecords(imgs, labels, n_classes, batch_size):
    imgs_batch, labels_batch = tf.train.shuffle_batch([imgs, labels], batch_size=batch_size, capacity=2000, min_after_dequeue=1000)
    labels_batch = tf.one_hot(labels_batch, n_classes)
    labels_batch = tf.cast(labels_batch, dtype=tf.int64)
    labels_batch = tf.reshape(labels_batch, [batch_size, n_classes])
    return imgs_batch, labels_batch              


def read_h5py():
    h5f = h5py.File('train.h5', 'r')
    data = h5f['my_datas']
    label = h5f['my_labels']
    print(label[15058])
    print(data.shape)
    cv2.imshow('ss', data[15058])
    cv2.waitKey(0)
    

def read_npy():
    x, y = np.load('X.npy'), np.load('Y.npy')
    print(x.shape)
    x = x.reshape(-1, 28, 28, 1)
    print(y[15058])
    print(x.shape)
    print(y.shape)
    cv2.imshow('ss', x[15058])
    cv2.waitKey(0)


def mnist_to_img():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist_data = input_data.read_data_sets('MNIST_data/')
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


  
#rename(r"train\1")
#create_folder(5)
     
# mnist_to_img()
create_mydataset()

#create_npy(28, 28, 10) 

#create_TFrecords('train',28, 28, 10)






#create_csv()

# create_h5py_batch(2, 28, 28, 10)  
#create_h5py(28, 28, 10)   
# read_h5py()


# read_npy()
