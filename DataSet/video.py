import cv2
import tensorflow as tf
import numpy as np
import PIL.Image as Image


def weight_variable(shape):  
    initial = tf.truncated_normal(shape, stddev=0.1)  #正态分布生成w
    return tf.Variable(initial)  
def bias_variable(shape):  
    initial = tf.constant(0.1, shape=shape)  
    return tf.Variable(initial)  

''''' 
卷积和池化，使用卷积步长为1（stride size）,0边距（padding size） 
池化用简单传统的2x2大小的模板做max pooling 
'''  
def conv2d(x, W):  
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')  
def max_pool_2x2(x):  
    return tf.nn.max_pool(x, ksize=[1,2,2,1],  
                        strides=[1,2,2,1], padding='SAME')  
x = tf.placeholder(tf.float32,[None, 784], name='input') #图像输入向量  
W = tf.Variable(tf.zeros([784,10]))  #权重，初始化值为全零  
b = tf.Variable(tf.zeros([10]))  #偏置，初始化值为全零  

#第一层卷积，由一个卷积接一个maxpooling完成，卷积在每个  
#5x5的patch中算出32个特征。  
#卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，  
#接着是输入的通道数目，最后是输出的通道数目。   
#而对于每一个输出通道都有一个对应的偏置量。  
W_conv1 = weight_variable([5,5,1,32])  
b_conv1 = bias_variable([32])  

'''''把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。 
'''  
x_image = tf.reshape(x, [-1,28,28,1])  #最后一维代表通道数目，如果是rgb则为3  
#x_image权重向量卷积，加上偏置项，之后应用ReLU函数，之后进行max_polling  
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)  
h_pool1 = max_pool_2x2(h_conv1)  

#实现第二层卷积  

#每个5x5的patch会得到64个特征  
W_conv2 = weight_variable([5, 5, 32, 64])  
b_conv2 = bias_variable([64])  

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
h_pool2 = max_pool_2x2(h_conv2)  


''''' 
图片尺寸变为7x7，加入有1024个神经元的全连接层，把池化层输出张量reshape成向量 
乘上权重矩阵，加上偏置，然后进行ReLU 
'''  
W_fc1 = weight_variable([7*7*64,1024])  
b_fc1 = bias_variable([1024])  

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  

#Dropout， 用来防止过拟合 #加在输出层之前，训练过程中开启dropout，测试过程中关闭  
keep_prob = tf.placeholder("float", name='keep_prob')  
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  

#输出层, 添加softmax层  
W_fc2 = weight_variable([1024,10])  
b_fc2 = bias_variable([10])  

y = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2, name='softmax')  









sess = tf.Session()
saver = tf.train.Saver()  
saver.restore(sess, "save/model.ckpt")




cap = cv2.VideoCapture(0)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        roi_X1=250
        roi_X2=300
        roi_Y1=200
        roi_Y2=250
        roi=frame[roi_Y1:roi_Y2, roi_X1:roi_X2]
        cv2.imshow('roi', roi)
        cv2.rectangle(frame,(roi_X1,roi_Y1),(roi_X2,roi_Y2),(255,0,0),2)
    
        gray = cv2.cvtColor(roi,  cv2.COLOR_BGR2GRAY)
        img=cv2.resize(gray ,(28,28))
        #img=np.array(img, dtype=np.uint8).reshape(1,784)
        img=((255-np.array(img, dtype=np.uint8)) /255.0).reshape(1,784)
        img_out_softmax=sess.run(y, feed_dict={x:img, keep_prob:1.0}).flatten().tolist()
        
        for i in img_out_softmax:
            if i>0.9:
                print(img_out_softmax.index(i))
                cv2.putText(frame, str(img_out_softmax.index(i)) ,(30,30) ,cv2.FONT_HERSHEY_COMPLEX,1,(20,100,255),3)
                cv2.putText(frame, str(i*100)[0:3]+'%' ,(70,30) ,cv2.FONT_HERSHEY_COMPLEX,1,(20,255,20),3)
        
        cv2.imshow('iframe', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()

cv2.destroyAllWindows()




img = cv2.imread("213.jpg",0)
img=cv2.resize(img ,(28,28))

img=((255-np.array(img, dtype=np.uint8)) /255.0).reshape(1,784)


print(sess.run(y, feed_dict={x:img, keep_prob:1}).flatten().tolist())

img_out_softmax=sess.run(y, feed_dict={x:img, keep_prob:1.0})
print(img_out_softmax)
prediction_labels = np.argmax(img_out_softmax, axis=1)
print(prediction_labels)






