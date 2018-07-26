# coding: utf-8
import cv2
import numpy as np
import sys
import glob
from pyzbar.pyzbar import decode
img_path=sys.argv[1]
model_img=sys.argv[2]
model_txt=sys.argv[3]
threshold=sys.argv[4]
rotate=sys.argv[5]
showbar=sys.argv[6]
black=sys.argv[7]
black_model=sys.argv[8]
black_threshold=sys.argv[9]

#with open('results.txt','a') as file:
imgs = glob.glob(img_path)
if black=='1':
    template =cv2.imdecode(np.fromfile(black_model,dtype=np.uint8),-1)
else:
    template =cv2.imdecode(np.fromfile(model_img,dtype=np.uint8),-1)
    
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
w2, h2 = template.shape[::-1]
    
for img_p in imgs:
    img=cv2.imdecode(np.fromfile(img_p,dtype=np.uint8),-1)
    if rotate=='1':
        img = cv2.flip(cv2.transpose(img), 1)
    elif rotate=='2':
        img=cv2.flip(img, -1)
    elif rotate=='3':
        img= cv2.flip(cv2.transpose(cv2.flip(img, -1)), 1)
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
   
    led = rotated[bottom_right2[1]+int(13*h2/48) : bottom_right2[1]+int(19*h2/48), top_left2[0] + int(16*w2/48): top_left2[0] + int(37*w2/96)]
    print(np.mean(led))
    if np.mean(led)>190:
        print("led_on"+' ',end="")
    else:
        print("led_off"+' ',end="")

    cv2.imshow("img",led )
    cv2.waitKey(0)
        
        
    if black=='1':
        w3, h3 = dst.shape[::-1]
       # std = np.std(dst[int(4*h3/23):int(17*h3/24), int(w3/10):int(9*w3/10) ])
        std = np.std(dst)
        if std>float(black_threshold):
            print("black_error"+' ',end="")
        else:
            print('black_normal'+' ',end="")
       # print(std)
        # cv2.imshow("img",dst)
       #cv2.imshow("img",dst[int(4*h3/23):int(17*h3/24), int(w3/10):int(9*w3/10) ])
        # cv2.waitKey(0)

    else:
        error=0
        str1=''
        with open(model_txt, 'r') as file2:
            error=0
            lines = file2.readlines()
            for line in lines:
                list = line.split()
                std = np.std(dst[int(list[1]):int(list[3]), int(list[0]):int(list[2])])
                if std < int(threshold):
                    cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,0,255), 1)
                    error=error+1
                    str1=str1+' '+str(list[4])
                else:
                    cv2.rectangle(roi,(int(list[0]),int(list[1])), (int(list[2]),int(list[3])), (0,255,0), 1)
        if error==0:
            print('normal ' +' ',end="")
            cv2.imshow("img",roi )
            cv2.waitKey(0)
        elif error>30:
            print('backScreen:' +' ',end="")
         # file.write('backScreen:'+img_p+'\n')
        else:
            print('error:'+str1 +' ',end="")
         # file.write('error:'+str1+'  path:'+img_p +'\n')
         
            # cv2.imshow("img",roi )
            # cv2.waitKey(0)
              
         
         
    if showbar=='1':
        bars=decode(gray[int(h1/2):int(h1),int(w1/5):int(w1)])
        if bars:
            for i in range(len(bars)):
                print(str(int(bars[i][0]))+'  path:'+img_p)
        else:
            print('no bars '+'  path:'+img_p)
