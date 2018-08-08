from PyQt5 import QtCore, QtGui, QtWidgets
from ui.set import Ui_MainWindow
import pymssql
from m_db import Db
from m_video import Video
import os
import binascii
import struct

class Set(QtWidgets.QMainWindow,Ui_MainWindow):
    
    def __init__(self):
        super(Set,self).__init__()
        self.setupUi(self)
        if os.path.isfile('config.txt'):
            with open('config.txt','r') as file:
                for line in file:
                    a = line.split()
                    if a[0] == 'ip':
                        self.lineEdit.setText(a[1].strip())
                    if a[0] == 'database':
                        self.lineEdit_2.setText(a[1].strip())
                    if a[0] == 'usr':
                        self.lineEdit_3.setText(a[1].strip())
                    if a[0] == 'pwd':
                        self.lineEdit_4.setEchoMode(QtWidgets.QLineEdit.Password)
                        self.lineEdit_4.setText(a[1].strip())
                    if a[0] == 'cam':
                        self.lineEdit_14.setText(a[1].strip()) 
                    if a[0] == 'imgpath':
                        self.lineEdit_15.setText(a[1].strip())     
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton1.clicked.connect(self.setDb)
        self.pushButton2.clicked.connect(self.setCOM)
        self.pushButton3.clicked.connect(self.setVideo)
        
        self.comboBox.addItems([port[0] for port in Video.checkCOM()])
        self.comboBox_3.addItem(' ')
        self.comboBox_3.addItems([port[0] for port in Video.checkCOM()])

  
    



    def setCOM(self):
        if Video.getFlagCOM():
            Video.closeCOM()
            self.pushButton2.setText('未连接')
            self.pushButton2.setStyleSheet('color:rgb(220, 70, 70);')
        else:
            try:
                Video.setCOM(self.comboBox.currentText(), self.comboBox_2.currentText())
            except :
                QtWidgets.QMessageBox().critical(self," ", '串口打开失败！')
                return
           # QtWidgets.QMessageBox().critical(self," ", '串口打开成功！')
            self.pushButton2.setText('已连接')
            self.pushButton2.setStyleSheet('color:rgb(0,240,0);')


    def openfile(self):
        openfile_name = QtWidgets.QFileDialog.getOpenFileName(self,'选择文件','','Txt files(*.txt )')
        if openfile_name[0] == '':
            return
        GNS=''
        IMU=''
        ATM=''
        UAV=''
        IMU_over=False
        GNS_over=False
        ATM_over=False
        UAV_over=False
        findGNS = 0
        findIMU = 0
        findATM = 0
        findUAV = 0
        all_GNS =0
        right_GNS =0
        all_IMU =0
        right_IMU =0
        all_ATM =0
        right_ATM =0
        all_UAV  =0
        right_UAV  =0

        name = openfile_name[0].split('.txt')[0]
        with open(openfile_name[0], 'r') as file:
            f_GNS = open(name+'_GNS.txt', 'w+')
            f_IMU = open(name+'_IMU.txt', 'w+')
            f_ATM = open(name+'_ATM.txt', 'w+')
            f_UAV = open(name+'_UAV.txt', 'w+')
            for line in file:
                hexlist = line.split()
            for mhex in hexlist:
                if findGNS ==0:
                    if mhex == '47':
                        findGNS+=1
                elif findGNS ==1:
                    if mhex == '4E':
                        findGNS+=1
                    else:
                        findGNS = 0
                elif findGNS ==2:
                    if mhex == '53':
                        findGNS+=1
                    else:
                        findGNS = 0
                elif findGNS ==3:
                    if mhex == '09':
                        findGNS+=1
                        all_GNS+=1
                    else:
                        findGNS = 0
                elif findGNS==4:
                    if len(GNS) <= 83:
                        try:
                            GNS+=bytes.decode(binascii.a2b_hex(mhex))
                        except:
                            findGNS=0
                            GNS=''
                            GNS_over=False
                        if len(GNS) == 82 and mhex=='0D':
                            GNS_over=True
                        if len(GNS) == 83 and mhex=='0A' and GNS_over:
                            right_GNS+=1
                            f_GNS.write(GNS)
                            findGNS=0
                            GNS=''
                            GNS_over=False
                    else:
                        findGNS=0
                        GNS_over=False
                        GNS=''

                    # try:
                    #     GNS+=bytes.decode(binascii.a2b_hex(mhex))
                    # except:
                    #     findGNS=0
                    #     over=0
                    #     GNS=''

                    # if over == 0:
                    #     if mhex == '0D':
                    #         over+=1
                    # elif over == 1:
                    #     if mhex == '0A':
                    #         print()
                    #         over=0
                    #         findGNS=0
                    #         if len(GNS)==83:
                    #             f_GNS.write(GNS)
                    #         GNS=''
                    #     else:
                    #         over=0

                if findIMU ==0:
                    if mhex == '49':
                        findIMU+=1
                elif findIMU ==1:
                    if mhex == '4D':
                        findIMU+=1
                    else:
                        findIMU = 0
                elif findIMU ==2:
                    if mhex == '55':
                        findIMU+=1
                    else:
                        findIMU = 0
                elif findIMU ==3:
                    if mhex == '09':
                        findIMU+=1
                        all_IMU+=1
                    else:
                        findIMU = 0
                elif findIMU==4:
                    if len(IMU) <= 147:
                        try:
                            IMU+=bytes.decode(binascii.a2b_hex(mhex))
                        except:
                            findIMU=0
                            IMU=''
                            IMU_over=False
                        if len(IMU) == 147 and mhex=='0D':
                            IMU_over=True
                        if len(IMU) == 148 and mhex=='0A' and IMU_over:
                            right_IMU+=1
                            f_IMU.write(IMU)
                            findIMU=0
                            IMU=''
                            IMU_over=False
                    else:
                        findIMU=0
                        IMU_over=False
                        IMU=''

                    # try:
                    #     IMU+=bytes.decode(binascii.a2b_hex(mhex))
                    # except:
                    #     findIMU=0
                    #     over=0
                    #     IMU=''
                    #     # print('erroe2')
                    # if len(IMU) > 149:
                    #     error333+=1
                    #     over=0
                    #     findIMU=0
                    #     IMU=''
                    # else:
                    #     if len(IMU) == 148:
                    #         mhex=='0D'
                    #     if over == 0:
                    #         if mhex == '0D':
                    #             over+=1
                    #     elif over == 1:
                    #         if mhex == '0A':
                    #             over=0
                    #             findIMU=0
                    #             if len(IMU) == 149:
                    #                 f_IMU.write(IMU)
                    #                 right_GNS+=1
                    #             else:
                    #                 error333+=1
                    #             IMU=''  
                    #         else:
                    #             over=0
                
                if findATM ==0:
                    if mhex == '41':
                        findATM+=1
                elif findATM ==1:
                    if mhex == '54':
                        findATM+=1
                    else:
                        findATM = 0
                elif findATM ==2:
                    if mhex == '4D':
                        findATM+=1
                    else:
                        findATM = 0
                elif findATM ==3:
                    if mhex == '09':
                        findATM+=1
                        all_ATM+=1
                    else:
                        findATM = 0
                elif findATM==4:
                    if len(ATM) <= 201:
                        ATM+=mhex+' '
                        if len(ATM) == 198 and mhex=='0D':
                            ATM_over=True
                        if len(ATM) == 201 and mhex=='0A' and ATM_over:
                            data = ATM[:-6].split()
                            if data[0]=='01' and data[1]=='04':
                                leng=int(data[2],16)
                                if len(data) != leng+5  or  (len(data)-5)%4 != 0:
                                    ATM=''
                                else:
                                    result=''
                                    data2=data[3:-2]
                                    for i in range(int(len(data2)/4)):
                                        ss = str(data2[i*4])+str(data2[i*4+1])+str(data2[i*4+2])+str(data2[i*4+3])
                                        if i >int(len(data2)/4)-3 :
                                            result+=str(struct.unpack('!f',binascii.unhexlify(ss))[0])+'  '
                                        else:
                                            result+=str(struct.unpack('!f',binascii.unhexlify(ss))[0])+'  '
                                    f_ATM.write(result+'\r\n')
                                    right_ATM+=1
                            findATM=0
                            ATM=''
                            ATM_over=False
                    else:
                        findATM=0
                        ATM_over=False
                        ATM=''

                    # ATM+=mhex+' '
                    # if len(ATM) ==198:
                    #     print(mhex)
                    # if over == 0:
                    #     if mhex == '0D':
                    #         over+=1
                    # elif over == 1:
                    #     if mhex == '0A':
                    #         over=0
                    #         findATM=0
                    #         data = ATM[:-6].split()
                    #         if data[0]=='01' and data[1]=='04':
                    #             leng=int(data[2],16)
                    #             if len(data) != leng+5  or  (len(data)-5)%4 != 0:
                    #                 ATM=''
                    #             else:
                    #                 result=''
                    #                 data2=data[3:-2]
                    #                 for i in range(int(len(data2)/4)):
                    #                     ss = str(data2[i*4])+str(data2[i*4+1])+str(data2[i*4+2])+str(data2[i*4+3])
                    #                     if i >int(len(data2)/4)-3 :
                    #                         result+=str(struct.unpack('!f',binascii.unhexlify(ss))[0])+'  '
                    #                     else:
                    #                         result+=str(struct.unpack('!f',binascii.unhexlify(ss))[0])+'  '
                    #                 f_ATM.write(result+'\r\n')
                    #                 ATM=''
                    #     else:
                    #         over=0

                if findUAV ==0:
                    if mhex == '55':
                        findUAV+=1
                elif findUAV ==1:
                    if mhex == '41':
                        findUAV+=1
                    else:
                        findUAV = 0
                elif findUAV ==2:
                    if mhex == '56':
                        findUAV+=1
                    else:
                        findUAV = 0
                elif findUAV ==3:
                    if mhex == '09':
                        findUAV+=1
                        all_UAV+=1
                    else:
                        findUAV = 0
                elif findUAV==4:
                    if len(IMU) <= 34:
                        UAV+=mhex
                        if len(UAV) == 32 and mhex=='0D':
                            UAV_over=True
                        if len(UAV) == 34 and mhex=='0A' and UAV_over:
                            f_UAV.write(UAV[2:-4]+'\r\n')
                            right_UAV+=1
                            findUAV=0
                            UAV=''
                            UAV_over=False
                    else:
                        findUAV=0
                        UAV_over=False
                        UAV=''

                    # UAV+=mhex
                    # if over == 0:
                    #     if mhex == '0D':
                    #         over+=1
                    # elif over == 1:
                    #     if mhex == '0A':
                    #         print(len(UAV))
                    #         over=0
                    #         findUAV=0
                    #         if len(UAV[2:-4]) == 30:
                    #             f_UAV.write(UAV[2:-4]+'\r\n')
                    #         UAV=''

                    #     else:
                    #         over=0

            f_GNS.close()
            f_IMU.close()
            f_ATM.close()
            f_UAV.close()
        QtWidgets.QMessageBox().information(self," ", '转化成功！\r\n\r\nGNS接收：'+str(all_GNS)+' 转化：'+str(right_GNS)+
                                                                   '\r\nIMU接收：'+str(all_IMU)+' 转化：'+str(right_IMU)+
                                                                   '\r\nATM接收：'+str(all_ATM)+' 转化：'+str(right_ATM)+
                                                                   '\r\nUAV接收：'+str(all_UAV)+' 转化：'+str(right_UAV))

                    




    def setDb(self):
        qmsgBox = QtWidgets.QMessageBox()
        if Db.getFlag():
            Db.discon()
            self.pushButton1.setText('未连接')
            self.pushButton1.setStyleSheet('color:rgb(220, 70, 70);')
        else:
            try:
                Db.m_con(self.lineEdit.text(),self.lineEdit_3.text(),self.lineEdit_4.text(), self.lineEdit_2.text())
                self.pushButton1.setText('已连接')
                self.pushButton1.setStyleSheet('color:rgb(0,240,0);')
                with open('config.txt','w+') as file:
                    file.write('ip '+self.lineEdit.text()  + '\n')
                    file.write('database '+self.lineEdit_2.text()+ '\n')
                    file.write('usr '+self.lineEdit_3.text()+ '\n')
                    file.write('pwd '+self.lineEdit_4.text()+ '\n')
                    file.write('cam '+self.lineEdit_14.text()  + '\n')
                    file.write('imgpath '+self.lineEdit_15.text()  + '\n')
            except :
                qmsgBox.critical(self," ", '数据库连接失败！')
          

    def setVideo(self, setVideo):
        if Video.getFlag():
            Video.closeCAM()
            self.pushButton3.setText('未连接')
            self.pushButton3.setStyleSheet('color:rgb(220, 70, 70);')
        else:
            result = Video.setCAM(self.lineEdit_14.text(), self.lineEdit_15.text())
            if result == 0:
                self.pushButton3.setText('已连接')
                self.pushButton3.setStyleSheet('color:rgb(0,240,0);')
                with open('config.txt','w+') as file:
                    file.write('ip '+self.lineEdit.text()  + '\n')
                    file.write('database '+self.lineEdit_2.text()+ '\n')
                    file.write('usr '+self.lineEdit_3.text()+ '\n')
                    file.write('pwd '+self.lineEdit_4.text()+ '\n')
                    file.write('cam '+self.lineEdit_14.text()  + '\n')
                    file.write('imgpath '+self.lineEdit_15.text()  + '\n')
            elif result == 1:
                QtWidgets.QMessageBox().critical(self," ", '视频源打开失败！')
            elif result == 2:
                QtWidgets.QMessageBox().critical(self," ", 'cnn.h5文件丢失！')





        
            
            
