import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from ui.Mainwindow import Ui_MainWindow
from m_video import Video
from m_set import Set
#from m_db import Db

class Main(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setWindowTitle("地面接收软件") 
        self.setupUi(self)
        self.resize(1366,768)
        self.showMaximized()

        self.child1 = Video()
        self.child3 = Set()
        self.pushButton_2.clicked.connect(self.Video)
        self.pushButton_3.clicked.connect(self.Set)
        #self.pushButton_4.clicked.connect(self.Home)
        self.Video()

    def Set(self):
       # self.gridLayout.removeWidget(self.label)
        self.gridLayout.addWidget(self.child3)
        self.child1.hide()
       # self.child2.hide()
        self.child3.show()


    def Home(self):
        self.gridLayout.addWidget(self.label)
        self.child1.hide()
        #self.child2.hide()
        self.child3.hide()


    def Video(self):
       # self.gridLayout.removeWidget(self.label)
        self.gridLayout.addWidget(self.child1)
        #self.child2.hide()
        self.child3.hide()
        self.child1.show()
        

    def Db(self):
        self.gridLayout.removeWidget(self.label)
        self.gridLayout.addWidget(self.child2)
        self.child1.hide()
        self.child3.hide()
        #self.child2.show()
    

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
        #    self.child2.close()
            self.child3.close()
            event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())




