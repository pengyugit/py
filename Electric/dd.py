import os
from PIL import Image
import numpy as np
import cv2
for i in range(81):
    path='train/'+str(i)
    if os.path.exists(path):
        filelist = os.listdir(path)  # 该文件夹下所有的文件（包括文件夹）
        for files in filelist:
  
            if str(files)=='76.jpg'  or str(files)=='430.jpg' \
            or str(files)=='434.jpg' or str(files)=='495.jpg' \
            or str(files)=='500.jpg' or str(files)=='503.jpg' \
            or str(files)=='571.jpg' or str(files)=='1615.jpg'\
            or str(files)=='1608.jpg' or str(files)=='1609.jpg' \
            or str(files)=='1610.jpg' or str(files)=='1613.jpg' \
            or str(files)=='1611.jpg' or str(files)=='1612.jpg' \
            or str(files)=='1616.jpg' or str(files)=='1687.jpg' \
            or str(files)=='567.jpg' or str(files)=='568.jpg' \
            or int(str(files)[:-4])>300:
                os.remove('train/'+str(i)+'/'+str(files))
                
##            img =Image.open('train/'+str(i)+'/'+str(files)).convert('L')
##            std = np.std(img)
##            if std<6:
##                print(str(std) +' train/'+str(i)+'/'+str(files))
##                cv2.imshow('s',np.array(img))
##                cv2.waitKey(0)
            
            #print(std)
            

print('over')
  
             
                
              
            
            

