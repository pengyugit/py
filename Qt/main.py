import sys
import cv2
from PyQt5 import QtCore, QtGui,QtWidgets

from Mainwindow import Ui_Form
from Video import Ui_Form1
from test import Ui_Form2

class Main(QtWidgets.QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setWindowTitle("MyExample")  ##设置窗口标题
        self.setupUi(self)

        self.child1 = Video()
        self.child2 = test()
        self.pushButton.clicked.connect(self.Video)
        self.pushButton_2.clicked.connect(self.test)

        
    def test(self):
        self.gridLayout.addWidget(self.child1)
        self.child2.close()
        self.child1.show()
        
    def Video(self):
        self.gridLayout.addWidget(self.child2)
        self.child1.close()
        self.child2.show()
    
    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cacel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")
        msg.addButton(ok,QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cacel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            self.child1.close()
            self.child2.close()
            event.accept()






    

class Video(QtWidgets.QMainWindow,Ui_Form1):
    def __init__(self):
        super(Video,self).__init__()
        self.setupUi(self)

        self.CAM_NUM = 0
        self.cap= cv2.VideoCapture()
        self.playTimer= QtCore.QTimer()
        
        self.openButton.clicked.connect(self.openButton_click)
        self.playTimer.timeout.connect(self.show_camera)

    def openButton_click(self):
        if self.playTimer.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确", 
                buttons=QtWidgets.QMessageBox.Ok,defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.playTimer.start(30)
                self.openButton.setText(u'关闭相机')
        else:
            self.playTimer.stop()
            self.cap.release()
            self.label_show_camera.clear()
            self.openButton.setText(u'打开相机')


    def show_camera(self):
        flag, self.frame = self.cap.read()
        frame = cv2.resize(self.frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(img))


    def closeEvent(self, event):
        self.label_show_camera.clear()
        self.openButton.setText(u'打开相机')
        if self.cap.isOpened():
            self.cap.release()
        if self.playTimer.isActive():
            self.playTimer.stop()
        event.accept()
               

 
class test(QtWidgets.QMainWindow,Ui_Form2):
    def __init__(self):
        super(test,self).__init__()
        self.setupUi(self)



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Main = Main()
    test = test()
    Video = Video()
    Main.show()
    sys.exit(app.exec_())




