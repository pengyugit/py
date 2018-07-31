
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
from keras.models import load_model
import os
from ui.Video import Ui_MainWindow

class Video(QtWidgets.QMainWindow,Ui_MainWindow):
    
    def __init__(self):
        super(Video,self).__init__()
        self.setupUi(self)

        #self.CAM_NUM = "rtsp://192.168.1.254:554/user=admin_password=tlJwpbo6_channel=1_stream=0.sdp?real_stream"
        self.CAM_NUM = '111.mp4'
        self.model = load_model('my_model.h5')
        self.cap= cv2.VideoCapture()
        self.playTimer= QtCore.QTimer()
        self.save_flag = False
        self.openButton.clicked.connect(self.openButton_click)
        self.openButton_2.clicked.connect(self.save_pic)
        self.openButton_2.setEnabled(False)
        self.playTimer.timeout.connect(self.show_camera)

    def openButton_click(self):
        if self.playTimer.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", 
                buttons=QtWidgets.QMessageBox.Ok,defaultButton=QtWidgets.QMessageBox.Ok)
                
            else:
                self.playTimer.start(1)
                self.openButton.setText(u'停止')
                self.openButton.setCheckable(True)
                self.openButton.setChecked(True)
                self.openButton_2.setEnabled(True)


        else:
            self.playTimer.stop()
            self.cap.release()
            self.label_show_camera.clear()
            _translate = QtCore.QCoreApplication.translate
            self.label_show_camera.setText(_translate("MainWindow", "视频显示区"))
            self.label.setText(_translate("MainWindow", "图像识别区"))
            self.save_flag = False
            self.openButton_2.setCheckable(False)
            self.openButton_2.setText(u'开始保存图片')
            self.openButton.setCheckable(False)
            self.openButton.setText(u'开始')
            self.openButton_2.setEnabled(False)


    def save_pic(self):
        if self.save_flag:
            self.save_flag = False
            self.openButton_2.setCheckable(False)
            self.openButton_2.setText(u'开始保存图片')
        else:
            if not os.path.exists(r'img'):
                    os.makedirs(r'img')
            self.save_flag = True
            self.openButton_2.setText(u'停止保存图片')
            self.openButton_2.setCheckable(True)
            self.openButton_2.setChecked(True)




    def show_camera(self):
        flag, img = self.cap.read()
        if flag:
            frame=img[1:15, 1:177]
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame = np.array(frame,np.float64) / 255.0
            c1=frame[:, 0:8]
            c2=frame[:, 8:16]
            c3=frame[:, 16:24]
            c4=frame[:, 24:32]
            c6=frame[:, 40:48]
            c7=frame[:, 48:56]
            c9=frame[:, 64:72]
            c10=frame[:, 72:80]
            c12=frame[:, 88:96]
            c13=frame[:, 96:104]
            c15=frame[:, 112:120]
            c16=frame[:, 120:128]
            c18=frame[:, 136:144]
            c19=frame[:, 144:152]
            c21=frame[:, 160:168]
            c22 = frame[:, 168:176]
            c1 = cv2.resize(c1, (25, 25))
            c1= c1.reshape([-1, 25, 25, 1])
            c2 = cv2.resize(c2, (25, 25))
            c2= c2.reshape([-1, 25, 25, 1])
            c3 = cv2.resize(c3, (25, 25))
            c3= c3.reshape([-1, 25, 25, 1])
            c4 = cv2.resize(c4, (25, 25))
            c4= c4.reshape([-1, 25, 25, 1])
            c6 = cv2.resize(c6, (25, 25))
            c6= c6.reshape([-1, 25, 25, 1])
            c7 = cv2.resize(c7, (25, 25))
            c7= c7.reshape([-1, 25, 25, 1])
            c9 = cv2.resize(c9, (25, 25))
            c9= c9.reshape([-1, 25, 25, 1])
            c10 = cv2.resize(c10, (25, 25))
            c10= c10.reshape([-1, 25, 25, 1])
            c12 = cv2.resize(c12, (25, 25))
            c12= c12.reshape([-1, 25, 25, 1])
            c13 = cv2.resize(c13, (25, 25))
            c13= c13.reshape([-1, 25, 25, 1])
            c15 = cv2.resize(c15, (25, 25))
            c15= c15.reshape([-1, 25, 25, 1])
            c16 = cv2.resize(c16, (25, 25))
            c16= c16.reshape([-1, 25, 25, 1])
            c18 = cv2.resize(c18, (25, 25))
            c18= c18.reshape([-1, 25, 25, 1])
            c19 = cv2.resize(c19, (25, 25))
            c19 = c19.reshape([-1, 25, 25, 1])
            c21 = cv2.resize(c21, (25, 25))
            c21 = c21.reshape([-1, 25, 25, 1])
            c22 = cv2.resize(c22, (25, 25))
            c22 = c22.reshape([-1, 25, 25, 1])
            C1=self.model.predict(c1)[0].tolist()
            r1 = C1.index(max(C1))
            C2=self.model.predict(c2)[0].tolist()
            r2 = C2.index(max(C2))
            C3=self.model.predict(c3)[0].tolist()
            r3 = C3.index(max(C3))
            C4=self.model.predict(c4)[0].tolist()
            r4 = C4.index(max(C4))
            C6=self.model.predict(c6)[0].tolist()
            r6 = C6.index(max(C6))
            C7=self.model.predict(c7)[0].tolist()
            r7 = C7.index(max(C7))
            C9=self.model.predict(c9)[0].tolist()
            r9 = C9.index(max(C9))
            C10=self.model.predict(c10)[0].tolist()
            r10 = C10.index(max(C10))
            C12=self.model.predict(c12)[0].tolist()
            r12 = C12.index(max(C12))
            C13=self.model.predict(c13)[0].tolist()
            r13 = C13.index(max(C13))
            C15=self.model.predict(c15)[0].tolist()
            r15 = C15.index(max(C15))
            C16=self.model.predict(c16)[0].tolist()
            r16 = C16.index(max(C16))
            C18=self.model.predict(c18)[0].tolist()
            r18 = C18.index(max(C18))
            C19=self.model.predict(c19)[0].tolist()
            r19 = C19.index(max(C19))
            C21=self.model.predict(c21)[0].tolist()
            r21 = C21.index(max(C21))
            C22=self.model.predict(c22)[0].tolist()
            r22 = C22.index(max(C22))
            name = str(r1)+str(r2)+str(r3)+str(r4)+'-' +str(r6)+str(r7)+'-' +str(r9)+str(r10)+'-' +str(r12)+str(r13)+'-'+str(r15)+str(r16)+'-'+str(r18)+str(r19)+'-'+str(r21)+str(r22)
            
            self.label.setText(name)
            if self.save_flag:
                if not os.path.exists(r'img/'+name+'.jpg'):
                    cv2.imwrite(r'img/'+name+'.jpg', img)

        
        # self.textBrowser.append()
            
            
            if self.comboBox.currentIndex() == 0:
                frame = cv2.resize(img, (640, 480))
            if self.comboBox.currentIndex() == 1:
                frame = cv2.resize(img, (1024, 768))
            if self.comboBox.currentIndex() == 2:
                frame = cv2.resize(img, (1920, 1080))

            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(img))
            self.label_show_camera.setScaledContents (True)
        else:
            self.cap.release()
            self.label_show_camera.clear()
            _translate = QtCore.QCoreApplication.translate
            self.label_show_camera.setText(_translate("MainWindow", "视频结束"))
            self.label.setText(_translate("MainWindow", "图像识别区"))
            self.save_flag = False
            self.openButton_2.setCheckable(False)
            self.openButton_2.setText(u'开始保存图片')
            self.openButton.setCheckable(False)
            self.openButton.setText(u'开始')
            self.openButton_2.setEnabled(False)
            self.playTimer.stop()


    def closeEvent(self, event):
        self.label_show_camera.clear()
        self.openButton.setText(u'打开相机')
        if self.cap.isOpened():
            self.cap.release()
        if self.playTimer.isActive():
            self.playTimer.stop()
        event.accept()