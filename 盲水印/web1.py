import cv2 
import numpy as np
from flask import Flask, make_response,Response,request,jsonify,render_template,session
import datetime
import base64

app = Flask(__name__)

@app.route('/encode', methods=['POST'])
def encode():
    arr = np.asarray(bytearray(request.files['image'].read()), dtype=np.uint8)
    arr2 = np.asarray(bytearray(request.files['wm'].read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)[:,:,:3]
    z1 = cv2.imdecode(arr2, -1)[:,:,:3]
    f1 = np.fft.fft2(img)
    zz = np.zeros(img.shape)
    h, w = img.shape[0], img.shape[1]
    hwm = np.zeros((int(h * 0.5), w, img.shape[2]))
    z1= cv2.resize(z1, (int(hwm.shape[1]/5), int(hwm.shape[0]/5)))
    for i in range(z1.shape[0]):
        for j in range(z1.shape[1]):
            #hwm[i][j] = z1[i][j]
            hwm[i+int(hwm.shape[0]/4)-1][j+int(hwm.shape[1]/4)-1] = z1[i][j]
    for i in range(hwm.shape[0]):
        for j in range(hwm.shape[1]):
            zz[i][j] = hwm[i][j]
            zz[zz.shape[0] - i - 1][zz.shape[1] - j - 1] = hwm[i][j]
    f2 = f1 +  zz
    f2 = np.fft.ifft2(f2)
    img_wm = np.real(f2)
    #img_wm = np.uint8(img_wm)
    cv2.imwrite('encode-'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S').strip()+'.png', img_wm)
    ret, png = cv2.imencode('.png', img_wm)
    return Response(png.tobytes(), status=200, mimetype='image/png')


@app.route('/decode', methods=['POST'])
def decode():
    arr = np.asarray(bytearray(request.files['image'].read()), dtype=np.uint8)
    img2 = cv2.imdecode(arr, -1)[:,:,:3]
    f3=np.fft.fft2(img2)
    f3 = np.real(f3)
    wm = np.zeros(f3.shape)
    for i in range(int(f3.shape[0] * 0.5)):
        for j in range(f3.shape[1]):
            wm[i][j] = np.uint8(f3[i][j])
    for i in range(int(f3.shape[0] * 0.5)):
        for j in range(f3.shape[1]):
            wm[f3.shape[0] - i - 1][f3.shape[1] - j - 1] = wm[i][j]
    cv2.imwrite('decode-'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S').strip()+'.png', wm)
    ret, png = cv2.imencode('.png', wm)
    return Response(png.tobytes(), status=200, mimetype='image/png')

@app.route('/decode2', methods=['POST'])
def decode2():
    arr = np.asarray(bytearray(request.files['image_wm'].read()), dtype=np.uint8)
    arr2 = np.asarray(bytearray(request.files['image'].read()), dtype=np.uint8)
    img2 = cv2.imdecode(arr, -1)[:,:,:3]
    img3 = cv2.imdecode(arr2, -1)[:,:,:3]
    f3=np.fft.fft2(img2)
    f4=np.fft.fft2(img3)
    f3 = np.real(f3-f4)
    wm = np.zeros(f3.shape)
    for i in range(int(f3.shape[0] * 0.5)):
        for j in range(f3.shape[1]):
            wm[i][j] = np.uint8(f3[i][j])
    for i in range(int(f3.shape[0] * 0.5)):
        for j in range(f3.shape[1]):
            wm[f3.shape[0] - i - 1][f3.shape[1] - j - 1] = wm[i][j]
    cv2.imwrite('decode-'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S').strip()+'.png', wm)
    ret, png = cv2.imencode('.png', wm)
    return Response(png.tobytes(), status=200, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')