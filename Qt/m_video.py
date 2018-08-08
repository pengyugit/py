import binascii
import os
import struct
from datetime import datetime

import cv2
import numpy as np
import serial
from keras.models import load_model
from PyQt5 import QtCore, QtGui, QtWidgets

from m_db import Db
from ui.Video import Ui_MainWindow

global CAM_NUM, cap, flag, model, imgPath, ser, flagcom, save_flag2
flag=False
flagcom=False
CAM_NUM = None
cap= cv2.VideoCapture()
model = None
save_flag2 = False
class Video(QtWidgets.QMainWindow,Ui_MainWindow):
    
    
    def __init__(self):
        super(Video,self).__init__()
        self.setupUi(self)
        self.playTimer= QtCore.QTimer()
        self.save_flag = False
        self.openButton.clicked.connect(self.openButton_click)
        self.openButton_2.clicked.connect(self.save_pic)
        self.openButton_2.setEnabled(False)
        self.playTimer.timeout.connect(self.show_camera)
        self.openButton_3.clicked.connect(self.openData)
        self.openButton_4.clicked.connect(self.save_data)
        self.openButton_4.setEnabled(False)
        self.mthread = mThread()    
        self.mthread.sinOut.connect(self.chuli)  
         

 
    def chuli(self, s):
        if 'GNS0' in s:
            self.label_2.setText('        '+str(s['GNS0'])+'         ')
        if 'GNS1' in s:
            self.label_3.setText('经度：'+str(s['GNS1']))
        if 'GNS2' in s:
            self.label_4.setText('纬度：'+str(s['GNS2']))
        if 'GNS3' in s:
            self.label_5.setText('高度：'+str(s['GNS3']))
        if s['all_GNS'] !=0:
            self.label_6.setText('GNS接收%d 质量：%.2f%%' % (s['all_GNS'], s['right_GNS']/s['all_GNS'] * 100))
        if s['all_IMU'] !=0:
            self.label_7.setText('IMU接收%d 质量：%.2f%%' % (s['all_IMU'], s['right_IMU']/s['all_IMU'] * 100)) 
        if s['all_ATM'] !=0:
            self.label_8.setText('ATM接收%d 质量：%.2f%%' % (s['all_ATM'], s['right_ATM']/s['all_ATM'] * 100))
        if s['all_UAV'] !=0:
            self.label_9.setText('UAV接收%d 质量：%.2f%%' % (s['all_UAV'], s['right_UAV']/s['all_UAV'] * 100))
        if 'db' in s:
            self.label_13.setText(str(s['db']))
  


    def openButton_click(self):
        global CAM_NUM, cap, flag
        if CAM_NUM == None:
            QtWidgets.QMessageBox().critical(self," ", '请先设置视频源！')
            return 
        if flag == False:
            self.playTimer.stop()
            QtWidgets.QMessageBox().critical(self," ", '请先设置视频源！')
            return
        if self.playTimer.isActive() == False:
            if cap.isOpened() == False:
                cap.open(CAM_NUM)
            self.playTimer.start(1)
            self.openButton.setText(u'停止')
            self.openButton.setCheckable(True)
            self.openButton.setChecked(True)
            self.openButton_2.setEnabled(True)
        else:
            cap.release()
            self.playTimer.stop()
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
            if not os.path.exists(imgPath):
                    os.makedirs(imgPath)
            self.save_flag = True
            self.openButton_2.setText(u'停止保存图片')
            self.openButton_2.setCheckable(True)
            self.openButton_2.setChecked(True)


    def save_data(self):
        global save_flag2
        if save_flag2:
            save_flag2 = False
            self.openButton_4.setCheckable(False)
            self.openButton_4.setText(u'开始保存数据')
        else:
            save_flag2 = True
            self.openButton_4.setText(u'停止保存数据')
            self.openButton_4.setCheckable(True)
            self.openButton_4.setChecked(True)



    def openData(self):
        global ser, flagcom, save_flag2
        if flagcom == False :
            QtWidgets.QMessageBox().critical(self," ", '请先设置串口！')
            return
        if self.mthread.isRunning()== False:
            self.mthread.start() 
            self.openButton_3.setText(u'停止')
            self.openButton_3.setCheckable(True)
            self.openButton_3.setChecked(True)
            self.openButton_4.setEnabled(True)
        else:
            self.mthread.terminate() 
            _translate = QtCore.QCoreApplication.translate
            self.label_2.setText(_translate("MainWindow", "时间"))
            self.label_3.setText(_translate("MainWindow", "经度"))
            self.label_4.setText(_translate("MainWindow", "纬度"))
            self.label_5.setText(_translate("MainWindow", "高度"))
            self.label_6.setText(_translate("MainWindow", "GNSS数据质量"))
            self.label_7.setText(_translate("MainWindow", "IMU数据质量"))
            self.label_8.setText(_translate("MainWindow", "大气数据质量"))
            self.label_9.setText(_translate("MainWindow", "吊舱数据质量"))
            save_flag2 = False
            self.openButton_4.setCheckable(False)
            self.openButton_4.setText(u'开始保存数据')
            self.openButton_3.setCheckable(False)
            self.openButton_3.setText(u'开始')
            self.openButton_4.setEnabled(False)


            

    

    def setCAM(path, img_path):
        if os.path.isfile('cnn.h5') == False:
            return 2
        global CAM_NUM, cap,model, flag, imgPath
        CAM_NUM = path
        if cap.open(CAM_NUM):
            model = load_model('cnn.h5')
            imgPath=img_path
            flag=True
            return 0
        else:
            return 1

    def closeCAM():
        global cap, flag
        cap.release()
        flag=False
        return True

    def getFlag():
        global  flag
        return flag

    def getFlagCOM():
        global  flagcom
        return flagcom

    def closeCOM():
        global ser,flagcom
        ser.close()
        flagcom=False
        return True
    
    def checkCOM():
        import serial.tools.list_ports
        return list(serial.tools.list_ports.comports())

    def setCOM(com, bot):
        global ser,flagcom
        ser = serial.Serial(com, int(bot))
        flagcom=True


    def show_camera(self):
        global cap, model
        flag, img = cap.read()
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
            C1=model.predict(c1)[0].tolist()
            r1 = C1.index(max(C1))
            C2=model.predict(c2)[0].tolist()
            r2 = C2.index(max(C2))
            C3=model.predict(c3)[0].tolist()
            r3 = C3.index(max(C3))
            C4=model.predict(c4)[0].tolist()
            r4 = C4.index(max(C4))
            C6=model.predict(c6)[0].tolist()
            r6 = C6.index(max(C6))
            C7=model.predict(c7)[0].tolist()
            r7 = C7.index(max(C7))
            C9=model.predict(c9)[0].tolist()
            r9 = C9.index(max(C9))
            C10=model.predict(c10)[0].tolist()
            r10 = C10.index(max(C10))
            C12=model.predict(c12)[0].tolist()
            r12 = C12.index(max(C12))
            C13=model.predict(c13)[0].tolist()
            r13 = C13.index(max(C13))
            C15=model.predict(c15)[0].tolist()
            r15 = C15.index(max(C15))
            C16=model.predict(c16)[0].tolist()
            r16 = C16.index(max(C16))
            C18=model.predict(c18)[0].tolist()
            r18 = C18.index(max(C18))
            C19=model.predict(c19)[0].tolist()
            r19 = C19.index(max(C19))
            C21=model.predict(c21)[0].tolist()
            r21 = C21.index(max(C21))
            C22=model.predict(c22)[0].tolist()
            r22 = C22.index(max(C22))
            name = str(r1)+str(r2)+str(r3)+str(r4)+'-' +str(r6)+str(r7)+'-' +str(r9)+str(r10)+'-' +str(r12)+str(r13)+'-'+str(r15)+str(r16)+'-'+str(r18)+str(r19)+'-'+str(r21)+str(r22)
            
            self.label.setText(name)
            if self.save_flag:
                if not os.path.exists(imgPath+'/'+name+'.jpg'):
                    cv2.imwrite(imgPath+'/'+name+'.jpg', img[18:, :])

        
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
            cap.release()
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
        global cap,ser
        self.label_show_camera.clear()
        self.openButton.setText(u'打开相机')
        if cap.isOpened():
            cap.release()
        # if ser.isOpen():
        #     ser.close()
        if self.playTimer.isActive():
            self.playTimer.stop()
        event.accept()





class mThread(QtCore.QThread):
    sinOut = QtCore.pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self.m = 0
        self.GNS=''
        self.IMU=''
        self.ATM=''
        self.UAV=''
        self.IMU_over=False
        self.GNS_over=False
        self.ATM_over=False
        self.UAV_over=False
        self.IMU_over1=False
        self.GNS_over1=False
      
        self.findGNS = 0
        self.findIMU = 0
        self.findATM = 0
        self.findUAV = 0
        self.all_GNS =0
        self.right_GNS =0
        self.all_IMU =0
        self.right_IMU =0
        self.all_ATM =0
        self.right_ATM =0
        self.all_UAV  =0
        self.right_UAV  =0
        self.saveGNS=''
        self.saveIMU=''
        self.saveATM=''
        self.saveUAV=''
        self.count_GNS=0
        self.count_IMU=0
        self.count_ATM=0
        self.count_UAV=0
        self.Db_GNS=[]
        self.Db_IMU=[]
        self.Db_ATM=[]
        
            
    def run(self):
        global ser,save_flag2 
        dic={}
        while 1:   
            if save_flag2==False :
                if self.saveGNS != '':
                    f_GNS = open('data_GNS.txt', 'w+')
                    f_GNS.write(self.saveGNS)
                    self.saveGNS=''
                    f_GNS.close()
                if self.saveIMU != '':
                    f_IMU = open('data_IMU.txt', 'w+')
                    f_IMU.write(self.saveIMU)
                    self.saveIMU=''
                    f_IMU.close()
                if self.saveATM != '':
                    f_ATM = open('data_ATM.txt', 'w+')
                    f_ATM.write(self.saveATM)
                    self.saveATM=''
                    f_ATM.close()
                if self.saveUAV != '':
                    f_UAV = open('data_UAV.txt', 'w+')
                    f_UAV.write(self.saveUAV)
                    self.saveUAV=''
                    f_UAV.close()
                  

            if ser.inWaiting()>0:
                data =ser.read(1)
                
                #mhex=bytes.decode(binascii.b2a_hex(data)).upper()
                mhex=data
                if self.findGNS ==0:
                    if mhex == b'G':
                        self.findGNS+=1
                elif self.findGNS ==1:
                    if mhex == b'N':
                        self.findGNS+=1
                    else:
                        self.findGNS = 0
                elif self.findGNS ==2:
                    if mhex == b'S':
                        self.findGNS+=1
                    else:
                        self.findGNS = 0
                elif self.findGNS ==3:
                    if mhex == b'\t':
                        self.findGNS+=1
                        self.all_GNS+=1
                    else:
                        self.findGNS = 0
                elif self.findGNS==4:
                    if mhex == b'\t':
                        self.count_GNS+=1
                    try:
                        self.GNS+=bytes.decode(mhex)
                    except:
                        self.findGNS=0
                        self.GNS=''
                        self.GNS_over=False
                        self.GNS_over1=False
                        self.count_GNS=0
                    if self.count_GNS == 14:
                        self.GNS_over1=True
                    if self.count_GNS > 14:
                        self.findGNS=0
                        self.GNS=''
                        self.GNS_over=False
                        self.GNS_over1=False
                        self.count_GNS=0
                    if mhex==b'\r'and self.GNS_over1:
                        self.GNS_over=True
                    if mhex==b'\n' and self.GNS_over and self.GNS_over1:
                        self.right_GNS+=1
                        list_gns=self.GNS.split()
                        gns_time = list_gns[0]+' '+list_gns[1]
                        gns_N = list_gns[4]
                        gns_E = list_gns[6]
                        gns_H = list_gns[8]
                        dic['GNS0']=gns_time
                        dic['GNS1']=gns_N
                        dic['GNS2']=gns_E
                        dic['GNS3']=gns_H
                        if save_flag2:
                            self.saveGNS+=self.GNS
                        if Db.getFlag():
                            gnsdata = ('001', gns_time , gns_N, gns_E, gns_H)
                            if gnsdata not in self.Db_GNS:
                                self.Db_GNS.append(gnsdata)
                            if len(self.Db_GNS)>100:
                                try:
                                    Db.insertGNSS(self.Db_GNS)
                                    self.Db_GNS=[]
                                    dic['db']= 'GNS上传成功 '+str(datetime.now().strftime('%H:%M:%S'))
                                except:
                                    dic['db']= 'GNS上传失败 '+str(datetime.now().strftime('%H:%M:%S'))
                                    self.Db_GNS=[]
                        self.findGNS=0
                        self.GNS=''
                        self.GNS_over=False
                        self.GNS_over1=False
                        self.count_GNS=0

        
                if self.findIMU ==0:
                    if mhex == b'I':
                        self.findIMU+=1
                elif self.findIMU ==1:
                    if mhex == b'M':
                        self.findIMU+=1
                    else:
                        self.findIMU = 0
                elif self.findIMU ==2:
                    if mhex == b'U':
                        self.findIMU+=1
                    else:
                        self.findIMU = 0
                elif self.findIMU ==3:
                    if mhex == b'\t':
                        self.findIMU+=1
                        self.all_IMU+=1
                    else:
                        self.findIMU = 0
                elif self.findIMU==4:
                    if mhex == b'\t':
                        self.count_IMU+=1
                    try:
                        self.IMU+=bytes.decode(mhex)
                    except:
                        self.findIMU=0
                        self.IMU=''
                        self.IMU_over=False
                        self.IMU_over1=False
                        self.count_IMU=0
                    if self.count_IMU == 23:
                        self.IMU_over1=True
                    if self.count_IMU > 23:
                        self.findIMU=0
                        self.IMU=''
                        self.IMU_over=False
                        self.IMU_over1=False
                        self.count_IMU=0
                    if mhex==b'\r'and self.IMU_over1:
                        self.IMU_over=True
                    if mhex==b'\n' and self.IMU_over and self.IMU_over1:
                        self.right_IMU+=1
                        list_imu=self.IMU.split()

                        if save_flag2:
                            self.saveIMU+=self.IMU
                        if Db.getFlag():
                            imudata = ('001', list_imu[0]+' '+list_imu[1] , list_imu[5], list_imu[3], list_imu[4], list_imu[11], list_imu[9], list_imu[10])
                            if imudata not in self.Db_IMU:
                                self.Db_IMU.append(imudata)
                            if len(self.Db_IMU)>150:
                                try:
                                    Db.insertIMU(self.Db_IMU)
                                    self.Db_IMU=[]
                                    dic['db']= 'IMU上传成功 '+str(datetime.now().strftime('%H:%M:%S'))
                                except:
                                    dic['db']= 'IMU上传失败 '+str(datetime.now().strftime('%H:%M:%S'))
                                    self.Db_IMU=[]
                        self.findIMU=0
                        self.IMU=''
                        self.IMU_over=False
                        self.IMU_over1=False
                        self.count_IMU=0


                if self.findATM ==0:
                    if mhex == b'A':
                        self.findATM+=1
                elif self.findATM ==1:
                    if mhex == b'T':
                        self.findATM+=1
                    else:
                        self.findATM = 0
                elif self.findATM ==2:
                    if mhex == b'M':
                        self.findATM+=1
                    else:
                        self.findATM = 0
                elif self.findATM ==3:
                    if mhex == b'\t':
                        self.findATM+=1
                        self.all_ATM+=1
                    else:
                        self.findATM = 0
                elif self.findATM==4:
                    mhex1=bytes.decode(binascii.b2a_hex(mhex)).upper()
                    if len(self.ATM) <= 201:
                        self.ATM+=mhex1+' '
                        if len(self.ATM) == 198 and mhex1=='0D':
                            self.ATM_over=True
                        if len(self.ATM) == 201 and mhex1=='0A' and self.ATM_over:
                            data = self.ATM[:-6].split()
                            if data[0]=='01' and data[1]=='04':
                                leng=int(data[2],16)
                                if len(data) != leng+5  or  (len(data)-5)%4 != 0:
                                    self.ATM=''
                                else:
                                    result=''
                                    data2=data[3:-2]
                                    for i in range(int(len(data2)/4)):
                                        ss = str(data2[i*4])+str(data2[i*4+1])+str(data2[i*4+2])+str(data2[i*4+3])
                                        if i >int(len(data2)/4)-3 :
                                            result+=str(struct.unpack('!f',binascii.unhexlify(ss))[0])+'  '
                                        else:
                                            result+=str(struct.unpack('!f',binascii.unhexlify(ss))[0])+'  '
                                    self.right_ATM+=1
                                    if save_flag2:
                                        self.saveATM+=result+'\r\n'
                                    list_atm=result.split()
                                    time_atm = list_atm[14].split('.')[0]
                                    year_time_atm = list_atm[13].split('.')[0]+' '+time_atm[:-4]+':'+time_atm[-4:-2]+':'+time_atm[-2:]
                                    atmdata = ('001', year_time_atm , list_atm[2], list_atm[11],list_atm[10], list_atm[9].split('.')[0], list_atm[8].split('.')[0], 
                                                list_atm[7].split('.')[0], list_atm[6], list_atm[5], list_atm[4], list_atm[3], list_atm[1], list_atm[0])
                                    if atmdata not in self.Db_ATM:
                                        self.Db_ATM.append(atmdata)
                                    if len(self.Db_ATM)==3:
                                        try:
                                            Db.insertATM(self.Db_ATM)
                                            self.Db_ATM=[]
                                            dic['db']= 'ATM上传成功 '+str(datetime.now().strftime('%H:%M:%S'))
                                        except:
                                            dic['db']= 'ATM上传失败 '+str(datetime.now().strftime('%H:%M:%S'))
                                            self.Db_ATM=[]

                            self.findATM=0
                            self.ATM=''
                            self.ATM_over=False
                    else:
                        self.findATM=0
                        self.ATM_over=False
                        self.ATM=''


                if self.findUAV ==0:
                    if mhex == b'U':
                        self.findUAV+=1
                elif self.findUAV ==1:
                    if mhex == b'A':
                        self.findUAV+=1
                    else:
                        self.findUAV = 0
                elif self.findUAV ==2:
                    if mhex == b'V':
                        self.findUAV+=1
                    else:
                        self.findUAV = 0
                elif self.findUAV ==3:
                    if mhex == b'\t':
                        self.findUAV+=1
                        self.all_UAV+=1
                    else:
                        self.findUAV = 0
                elif self.findUAV==4:
                    mhex2=bytes.decode(binascii.b2a_hex(mhex)).upper()
                    if len(self.UAV) <= 34:
                        self.UAV+=mhex2
                        if len(self.UAV) == 32 and mhex2=='0D':
                            self.UAV_over=True
                        if len(self.UAV) == 34 and mhex2=='0A' and self.UAV_over:
                            if save_flag2:
                                self.saveUAV+=self.UAV[:-4]
                           

                            self.right_UAV+=1
                            self.findUAV=0
                            self.UAV=''
                            self.UAV_over=False
                    else:
                        self.findUAV=0
                        self.UAV_over=False
                        self.UAV=''
             
                dic['right_GNS']=self.right_GNS
                dic['all_GNS']=self.all_GNS
                dic['right_IMU']=self.right_IMU
                dic['all_IMU']=self.all_IMU
                dic['right_ATM']=self.right_ATM
                dic['all_ATM']=self.all_ATM
                dic['right_UAV']=self.right_UAV
                dic['all_UAV']=self.all_UAV
          
                self.sinOut.emit(dic)    
