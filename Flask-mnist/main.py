import numpy as np
#import tensorflow as tf
from flask import Flask, make_response,Response,request,jsonify
import cv2
import base64
#from mnist import model1
from keras.models import load_model
from PIL import Image  


# webapp
app = Flask(__name__)



model_keras = load_model('mnist/data/my_model.h5')
img = Image.open('1.png').convert('L')
img = img.resize((28, 28))
img = 1-np.array(img) / 255.0
img= img.reshape([-1, 28, 28, 1])
output1=model_keras.predict(img)[0]



# x = tf.placeholder("float", [None, 784])
# sess = tf.Session()

# with tf.variable_scope("regression"):
    # y1, variables = model1.regression(x)
# saver = tf.train.Saver(variables)
# saver.restore(sess, "mnist/data/regression.ckpt")


# with tf.variable_scope("convolutional"):
    # keep_prob = tf.placeholder("float")
    # y2, variables = model.convolutional(x, keep_prob)
# saver = tf.train.Saver(variables)
# saver.restore(sess, "mnist/data/convolutional.ckpt-9999")


# def regression(input):
    # return sess.run(y1, feed_dict={x: input}).flatten().tolist()


# def convolutional(input):
    # return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()




@app.route('/api/mnist', methods=['POST'])
def mnist():



    # input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    # output1 = regression(input)
    output1 =[2.1323139378574374e-19, 1.0528872695578229e-12, 3.6811585680737724e-13, 1.0, 9.336113611667146e-16, 6.388241047261545e-09, 1.6650652069302875e-16, 5.365053887262938e-12, 1.335840882354944e-10, 7.14361958475962e-10]
    #output2 = convolutional(input)
    
    
    
    X = (1- np.array(request.json, dtype=np.uint8)/ 255.0).reshape([-1, 28, 28, 1])
    output3=model_keras.predict(X)[0].tolist()


    return jsonify(results=[output1,output3 ])

    
    
@app.route("/cat_dog",methods=['POST'])
def uploa222():
    # convnet = input_data(shape=[None, 50, 50, 1], name='input')
    # convnet = conv_2d(convnet, 32, 5, activation='relu')
    # convnet = max_pool_2d(convnet, 5)
    # convnet = conv_2d(convnet, 64, 5, activation='relu')
    # convnet = max_pool_2d(convnet, 5)
    # convnet = conv_2d(convnet, 128, 5, activation='relu')
    # convnet = max_pool_2d(convnet, 5)
    # convnet = conv_2d(convnet, 64, 5, activation='relu')
    # convnet = max_pool_2d(convnet, 5)
    # convnet = conv_2d(convnet, 32, 5, activation='relu')
    # convnet = max_pool_2d(convnet, 5)
    # convnet = fully_connected(convnet, 1024, activation='relu')
    # convnet = dropout(convnet, 0.8)
    # convnet = fully_connected(convnet, 2, activation='softmax')
    # model3 = tflearn.DNN(convnet)
    # model3.load('mnist/data/cat_dog')
    # if request.method == 'POST':
        # file = request.files['file']
    # file.save("static/temp.jpg")
    # img = cv2.imread('static/temp.jpg',0)
    # img=cv2.resize(img,(50,50))
    # img=np.array(img, dtype=np.uint8).reshape(-1, 50, 50, 1)
    # output1=model3.predict(img)[0][0].tolist()
    # output2=model3.predict(img)[0][1].tolist()
    # print(output1)
    return jsonify(results=[output1,output2])
    

@app.route("/upload",methods=['POST'])
def uploa2d():
    if request.method == 'POST':
        file = request.files['file']
    file.save("static/temp.jpg")
    img = cv2.imread('static/temp.jpg',0)
    #img=cv2.resize(img,(28,28))
    ret, jpeg = cv2.imencode('.jpg', img)
    return base64.b64encode(jpeg.tobytes())
    # input = ((255 - np.array(img, dtype=np.uint8)) / 255.0).reshape(1, 784)
    # output1 = regression(input)
    # output2 = convolutional(input)
    # print(output2)
    # return jsonify(results=[output1, output2])
    

@app.route("/image", methods=['POST','GET']) 
def index(): 
    img = cv2.imread('static/temp.jpg')
    img=cv2.resize(img,(600,800))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, jpeg = cv2.imencode('.jpg', img)
    return base64.b64encode(jpeg.tobytes())
    
@app.route("/image2", methods=['POST','GET']) 
def index2(): 
    img = cv2.imread('static/temp.jpg')
    img=cv2.resize(img,(600,800))
    ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret, jpeg = cv2.imencode('.jpg', img)
    return base64.b64encode(jpeg.tobytes())
    
    
    
@app.route("/image3", methods=['POST','GET']) 
def index3(): 
    img = cv2.imread('static/temp.jpg')
    img=cv2.resize(img,(600,800))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.Canny(img, 100, 255, 3) 
    ret, jpeg = cv2.imencode('.jpg', img)
    return base64.b64encode(jpeg.tobytes())
    

if __name__ == '__main__':
    app.run(host='0.0.0.0')
