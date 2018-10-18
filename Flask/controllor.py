import numpy as np
from flask import Flask, make_response,Response,request,jsonify,render_template,session
import cv2
import base64
import pymysql
from keras.models import load_model

#先预测一次防止keras报错
global img
#model = load_model('static/mnist.h5')
#img = cv2.imread('static/img/2.png', 0)
#img = cv2.resize(img, (28, 28))
#img = (1 - np.array(img) / 255.0).reshape([-1, 28, 28, 1])
#model.predict(img)


app = Flask(__name__)
app.config['SECRET_KEY'] = '\xf1\x92Y\xdf\x8ejY\x04\x96\xb4V\x88\xfb\xfc\xb5\x18F\xa3\xee\xb9\xb9t\x01\xf0\x96'
conn = pymysql.connect(host='127.0.0.1', user='root', password='123456', db='mydb', charset='utf8')
cur = conn.cursor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/register", methods=['POST'])
def register():
    data=request.get_json('data')
    name = data["name"] 
    id1 = data["id"] 
    password1 = data["password1"]
    password2 = data["password2"]
    major = data["major1"]
    line2="{\"result\":\""  
    if name=="" or id1=="" or password1=="" or password2==""or major=="":
        line2=line2+"注册信息不完整，请重新输入"
        return line2+"\"}" 
    if password1!= password2:
        line2=line2+"两次密码不一致，请重新输入" 
        return line2+"\"}" 
    if len(password1)<6:
        line2=line2+"密码长度至少6位" 
        return line2+"\"}" 
    sql = "SELECT id FROM student where id= "+id1
    cur.execute(sql)
    if cur.fetchone() is not None:
        return line2+"学号已经注册，请登陆\"}"
    sql = "insert into student (id,name,password,major) values ('%s','%s','%s','%s')"%(id1,name,password1,major)  
    # sql = "UPDATE EMPLOYEE SET AGE = AGE + 1 WHERE SEX = '%c'" % ('M')        
    # sql = "DELETE FROM EMPLOYEE WHERE AGE > '%d'" % (20)
    cur.execute(sql)
    conn.commit()
    return line2+"注册成功！\"}"

@app.route('/login', methods=['POST'])
def login():
    data=request.get_json('data')
    sql = "SELECT * FROM student where id= "+data['user']
    try:
        cur.execute(sql)
        result = cur.fetchone() # cursor.fetchall()
        if result[1] == data['pass']:
            session['username']=result[2]
            session['id']=result[0]
            session['major']=result[5]
            session['time']=result[7]
            return jsonify({'name': 'xmr', 'age': 18})
        else:
            return 'error'
    except:
        return 'error'

@app.route("/logout")
def logout():
    session.pop('username',None)
    session.pop('password',None)
    return render_template('index.html') 

@app.route('/home')
def home():
    if session.get('username') is not None:
        context={
            'name':session.get('username'),
            'id': session.get('id'),
            'major': session.get('major'),
            'time': session.get('time'),
            'renwu': "s",
        }
        return render_template('home.html', **context)
    else:
        return render_template('index.html')
       
@app.route("/listpeople")
def listpeople():
    if session.get('username') is not None:
        return render_template('listpeople.html',name=session.get('username')) 
    else:
        return render_template('index.html')


@app.route("/study_basics")
def study_basics():
    if session.get('username') is not None:
        return render_template('study_basics.html',name=session.get('username')) 
    else:
        return render_template('index.html')
@app.route("/study_imgprc")
def study_imgprc():
    if session.get('username') is not None:
        return render_template('study_imgprc.html',name=session.get('username')) 
    else:
        return render_template('index.html')
@app.route("/study_objdec")
def study_objdec():
    if session.get('username') is not None:
        return render_template('study_objdec.html',name=session.get('username')) 
    else:
        return render_template('index.html')
@app.route("/study_3d")
def study_3d():
    if session.get('username') is not None:
        return render_template('study_3d.html',name=session.get('username')) 
    else:
        return render_template('index.html')
@app.route("/imgProc")
def imgProc():
    if session.get('username') is not None:
        return render_template('imgProc.html',name=session.get('username')) 
    else:
        return render_template('index.html')
@app.route("/objdec")
def objdec():
    if session.get('username') is not None:
        return render_template('objdec.html',name=session.get('username')) 
    else:
        return render_template('index.html')

@app.route("/calibration")
def calibration():
    if session.get('username') is not None:
        return render_template('calibration1.html',name=session.get('username')) 
    else:
        return render_template('index.html')
@app.route("/measure")
def measure():
    if session.get('username') is not None:
        return render_template('measure.html',name=session.get('username')) 
    else:
        return render_template('index.html')
@app.route("/down")
def down():
    if session.get('username') is not None:
        return render_template('down.html',name=session.get('username')) 
    else:
        return render_template('index.html')
@app.route("/mnist1")
def mnist1():
    if session.get('username') is not None:
        return render_template('mnist.html',name=session.get('username')) 
    else:
        return render_template('index.html')



@app.route('/paper')
def paper():
    context=(
        ('name1','201622362013271','name','07','name','name','5','1'),
        ('name2','201622362013271','name','07','name','name','5','2'),
    )
    return render_template('paper.html', u=context)




@app.route('/api/mnist', methods=['POST'])
def mnist():
    X = (1- np.array(request.json, dtype=np.uint8)/ 255.0).reshape([-1, 28, 28, 1])
    output=model.predict(X)[0].tolist()
    return jsonify(results=[output])


@app.route("/upload",methods=['POST'])
def uploa2d():
    global img
    arr = np.asarray(bytearray(request.files['file'].read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img=cv2.resize(img,(100,100))
    ret, jpeg = cv2.imencode('.jpg', img)
    return base64.b64encode(jpeg.tobytes())

@app.route("/gray")
def gray():
    global img
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, jpeg = cv2.imencode('.jpg', img)
    return base64.b64encode(jpeg.tobytes())
    
@app.route("/canny")
def canny():
    global img
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.Canny(img, 100, 255, 3) 
    ret, jpeg = cv2.imencode('.jpg', img)
    return base64.b64encode(jpeg.tobytes())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
