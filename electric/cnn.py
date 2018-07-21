# coding: utf-8
import cv2
import numpy as np
import sys
import glob
from pyzbar.pyzbar import decode
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import load_csv

convnet = input_data(shape=[None, 28, 28, 1], name='input')
convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = dropout(convnet, 0.25)
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = dropout(convnet, 0.25)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.5)
convnet = fully_connected(convnet, 81, activation='softmax')
model2 = tflearn.DNN(convnet)
model2.load('my_model.tflearn')

img_path= 'img/*308.jpg'
model_img='单相模板.jpg'
model_txt='单相模板.txt'



imgs = glob.glob(img_path)

template =cv2.imdecode(np.fromfile(model_img,dtype=np.uint8),-1)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
w2, h2 = template.shape[::-1]
    
for img_p in imgs:
    print(img_p)
    img=cv2.imdecode(np.fromfile(img_p,dtype=np.uint8),-1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w1, h1 = img.shape[:2]
    center = (h1 // 2, w1 // 2)

    if h2>700:
        l=0.2
    elif h2>500:
        l=0.5
    elif h2>300:
        l=0.7
    else:
        l=0.8
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE, l)
    lines = lsd.detect(gray)
    for line in lines[0]:
        if np.abs(line[0][0] - line[0][2])>(4*w2/5) and np.abs(line[0][1] - line[0][3])<(h2/10):
            line2=line
            break
    angle = np.arctan((float)(line2[0][3] - line2[0][1]) / (float)(line2[0][2] - line2[0][0])) * 180 / np.pi
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (h1, w1))
    gray2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, _, _, top_left2 = cv2.minMaxLoc(cv2.matchTemplate(gray2, template, cv2.TM_CCOEFF_NORMED))
    bottom_right2 = (top_left2[0] + w2, top_left2[1] + h2)
    roi = rotated[top_left2[1]:bottom_right2[1], top_left2[0]:bottom_right2[0]]
    dst = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
  
    error=0
    str1=''
    with open(model_txt, 'r') as file2:
        error=0
        lines = file2.readlines()
        for line in lines:
            list = line.split()

            cnn_roi=cv2.resize(dst[int(list[1]):int(list[3]), int(list[0]):int(list[2])],(28,28))
            cnn_roi=np.array(cnn_roi, dtype=np.uint8).reshape(-1, 28, 28, 1)
            cnn_roi = 1 - cnn_roi/ 255.0  
            output1=model2.predict(cnn_roi)[0]
            output1 = output1.tolist()
            result = output1.index(max(output1))
            print(max(output1))
            if int(list[4]) <= 51 :
                if int(result) != int(list[4]):
                    cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,0,255), 1)
                    error=error+1
                    str1=str1+' '+str(list[4])
            elif int(list[4]) > 51 and int(list[4]) <= 59:
                if int(result) != 5:
                    cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,0,255), 1)
                    error=error+1
                    str1=str1+' '+str(5)
            elif int(list[4]) > 59 and  int(list[4]) < 85:
                if int(result) != (int(list[4])-8):
                    cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,0,255), 1)
                    error=error+1
                    str1=str1+' '+str(int(list[4])-8)
            elif int(list[4]) == 85:
                if int(result) != 32:
                    cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,0,255), 1)
                    error=error+1
                    str1=str1+' '+str(32)
            elif int(list[4]) == 86:
                if int(result) != 77:
                    cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,0,255), 1)
                    error=error+1
                    str1=str1+' '+str(77)
            elif int(list[4]) == 87:
                if int(result) != 78:
                    cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,0,255), 1)
                    error=error+1
                    str1=str1+' '+str(78)
            elif int(list[4]) == 88:
                if int(result) != 29:
                    cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,0,255), 1)
                    error=error+1
                    str1=str1+' '+str(29)
            elif int(list[4]) == 89:
                if int(result) != 30:
                    cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,0,255), 1)
                    error=error+1
                    str1=str1+' '+str(30)
            elif int(list[4]) == 90:
                if int(result) != 79:
                    cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,0,255), 1)
                    error=error+1
                    str1=str1+' '+str(79)
            elif int(list[4]) == 91:
                if int(result) != 80:
                    cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,0,255), 1)
                    error=error+1
                    str1=str1+' '+str(80)





            
            # std = np.std(dst[int(list[1]):int(list[3]), int(list[0]):int(list[2])])
            # if std < int(5):
            #     cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,0,255), 1)
            #     error=error+1
            #     str1=str1+' '+str(list[4])
            # else:
            #     cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,255,0), 1)
    if error==0:
        print('normal ' )
        cv2.imshow("img",roi )
        cv2.waitKey(0)
    elif error>30:
        print('backScreen:')
        cv2.imshow("img",roi )
        cv2.waitKey(0)
    else:
        print('error:'+str1 )
        cv2.imshow("img",roi )
        cv2.waitKey(0)
