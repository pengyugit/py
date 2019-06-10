import cv2 
import numpy as np
import sys


if len(sys.argv)<4:
    print('输入参数不正确')
    print('encode usage：python test1.py   encode   src.png  encode_img.png   wm.png')
    print('decode usage：python test1.py   decode   encode_img.png  dst_wm.png  ')
    sys.exit() 

order=sys.argv[1]
src=sys.argv[2]
dst=sys.argv[3]

if order == 'encode':
    if len(sys.argv)<5:
        print('输入参数不正确')
        print('encode usage：python test1.py   encode   src.png  encode_img.png   wm.png')
        sys.exit() 
    wm=sys.argv[4]

    try:
        img = cv2.imread(src)
        f1 = np.fft.fft2(img)
    except :
        print('输入图片路径错误')
        sys.exit() 
    
    zz = np.zeros(img.shape)
    z1= cv2.imread(wm)

    h, w = img.shape[0], img.shape[1]
    hwm = np.zeros((int(h * 0.5), w, img.shape[2]))

    z1= cv2.resize(z1, (int(hwm.shape[1]/5), int(hwm.shape[0]/5)))
    for i in range(z1.shape[0]):
        for j in range(z1.shape[1]):
            # hwm[i][j] = z1[i][j]
            hwm[i+int(hwm.shape[0]/4)-1][j+int(hwm.shape[1]/4)-1] = z1[i][j]
    #hwm = cv2.copyMakeBorder(z1, 100, int(hwm.shape[0]  - z1.shape[0]), 100, int(hwm.shape[1] - z1.shape[1]) , cv2.BORDER_CONSTANT, value=[0,0,0])
    # cv2.imshow('s',hwm)
    # cv2.waitKey(0)


    for i in range(hwm.shape[0]):
        for j in range(hwm.shape[1]):
            zz[i][j] = hwm[i][j]
            zz[zz.shape[0] - i - 1][zz.shape[1] - j - 1] = hwm[i][j]
    f2 = f1 +  zz

    f2 = np.fft.ifft2(f2)
    img_wm = np.real(f2)
    #img_wm = np.uint8(img_wm)
    cv2.imwrite(dst, img_wm)

elif order == 'decode':

    try:
        img2 = cv2.imread(src)
        f3=np.fft.fft2(img2)
    except :
        print('输入图片路径错误')
        sys.exit() 
    f3 = np.real(f3)
    

    wm = np.zeros(f3.shape)
    for i in range(int(f3.shape[0] * 0.5)):
        for j in range(f3.shape[1]):
            wm[i][j] = np.uint8(f3[i][j])
    for i in range(int(f3.shape[0] * 0.5)):
        for j in range(f3.shape[1]):
            wm[f3.shape[0] - i - 1][f3.shape[1] - j - 1] = wm[i][j]

    cv2.imwrite(dst, wm)
else:
    print('输入指令不正确（encode/decode）')
    print('encode usage：python test1.py   encode   src.png  encode_img.png   wm.png')
    print('decode usage：python test1.py   decode   encode_img.png  dst_wm.png  ')

