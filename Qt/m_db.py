
from PyQt5 import QtCore, QtGui, QtWidgets
from ui.database import Ui_MainWindow
import PyQt5.QtSql as sql

class Db(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(Db,self).__init__()
        self.setupUi(self)
        self.server = '192.168.50.177'
        self.database = 'db_UAVMS'
        self.user  = 'sa'
        self.password = 'ywg912918'

        # db=sql.QSqlDatabase.addDatabase(self.server)
        # db.setDatabaseName(self.database)
        # db.setUserName(self.user)
        # db.setPassword(self.password)
        # db.open()

        # query=sql.QSqlQuery('select * from GNSS')
        # query.first()
        # for i in range(query.size()):
        #     print(query.value(0),query.value(1))
        #     query.next()

        # db.close()

        # self.model=sql.QSqlQueryModel(self)
        # self.model.setHeaderData(0, QtCore.Qt.Horizontal, '名字1')
        # self.model.setHeaderData(1, QtCore.Qt.Horizontal, '名字')
        # self.model.setHeaderData(2, QtCore.Qt.Horizontal, '数量')
        
        # self.tableView.setModel(self.model)


        self.model=QtGui.QStandardItemModel(4,4)
        self.model.setHorizontalHeaderLabels(['标题1','标题2','标题3','标题4'])
        for row in range(4):
            for column in range(4):
                item = QtGui.QStandardItem(" %s,  %s"%(row,column))
                self.model.setItem(row, column, item)
       # self.tableView=QTableView()
        self.tableView.setModel(self.model)
  










# # 新建、插入操作
# cursor.execute("""
# IF OBJECT_ID('persons', 'U') IS NOT NULL
#     DROP TABLE persons
# CREATE TABLE persons (
#     id INT NOT NULL,
#     name VARCHAR(100),
#     salesrep VARCHAR(100),
#     PRIMARY KEY(id)
# )
# """)
# cursor.executemany(
#     "INSERT INTO persons VALUES (%d, %s, %s)",
#     [(1, 'John Smith', 'John Doe'),
#      (2, 'Jane Doe', 'Joe Dog'),
#      (3, 'Mike T.', 'Sarah H.')])
# # 如果没有指定autocommit属性为True的话就需要调用commit()方法
# conn.commit()
