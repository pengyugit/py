import numpy as np
#import tensorflow as tf
from flask import Flask, make_response,Response,request,jsonify,render_template
import cv2
import base64
#from mnist import model1
#from keras.models import load_model
from PIL import Image  


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
    output=model_keras.predict(X)[0].tolist()
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
    #app.run(host='0.0.0.0')
    app.run(debug=True, host='0.0.0.0')
