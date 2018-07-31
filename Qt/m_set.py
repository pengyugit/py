from PyQt5 import QtCore, QtGui, QtWidgets
from ui.set import Ui_MainWindow


class Set(QtWidgets.QMainWindow,Ui_MainWindow):
    
    def __init__(self):
        super(Set,self).__init__()
        self.setupUi(self)