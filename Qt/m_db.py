import pymssql

class Db():
    global conn ,flag
    flag=False
    def __init__(self):
        super(Db,self).__init__()
        self.setupUi(self)
    

    def m_con( ip, user, pwd, database):
        global conn ,flag 
        conn = pymssql.connect(ip, user, pwd, database, timeout=5, login_timeout=3)
        cursor = conn.cursor()
        flag=True
 

    def discon():
        global conn ,flag 
        conn.close()
        flag=False
        
    
    def getFlag():
        return flag


    def insertGNSS(GNSS):
        global conn 
        cursor = conn.cursor()
        cursor.executemany("INSERT INTO GNSS VALUES (%s, %s, %s, %s, %s)",GNSS)
        conn.commit()


    def insertIMU(IMU):
        global conn 
        cursor = conn.cursor()
        cursor.executemany("INSERT INTO IMU VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",IMU)
        conn.commit()
       

    def insertATM(ATM):
        global conn 
        cursor = conn.cursor()
        cursor.executemany("INSERT INTO  Atmosphere VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",ATM)
        conn.commit()
        





  

