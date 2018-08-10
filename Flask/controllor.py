import numpy as np
from flask import Flask, make_response,Response,request,jsonify,render_template
import cv2
import base64

# from keras.models import load_model
# #先预测一次防止keras报错
# model = load_model('static/mnist.h5')
# img = cv2.imread('static/img/2.png', 0)
# img = cv2.resize(img, (28, 28))
# img = (1 - np.array(img) / 255.0).reshape([-1, 28, 28, 1])
# model.predict(img)[0].tolist()


# webapp
app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html', name='py',id='201622362013271',major='自动化',time='2018-04-26 15:07:27',renwu='')











@app.route('/api/mnist', methods=['POST'])
def mnist():
    X = (1- np.array(request.json, dtype=np.uint8)/ 255.0).reshape([-1, 28, 28, 1])
    output=model.predict(X)[0].tolist()
    return jsonify(results=[output])

@app.route("/upload",methods=['POST'])
def uploa2d():
    arr = np.asarray(bytearray(request.files['file'].read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img=cv2.resize(img,(100,100))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.Canny(img, 100, 255, 3) 
    ret, jpeg = cv2.imencode('.jpg', img)
    return base64.b64encode(jpeg.tobytes())

    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
