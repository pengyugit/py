import sys

from PyQt5 import QtWidgets


import cv2

from PyQt5.QtGui import QImage, QPixmap

from my_ui import Ui_Form





class MainWindow(QtWidgets.QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("MyExample")  ##设置窗口标题

        self.setupUi(self)
        self.pushButton.setText("hello Python")
        self.pushButton.clicked.connect(self.onWorldClicked)


    def onWorldClicked(self, remark):
        print(remark)
        self.pushButton.setText("Hello World")
        img=cv2.imread('1.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(showImage))







if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
