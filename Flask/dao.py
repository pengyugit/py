from pymysql import connect
 
class Dao:
     
    def __init__(self):
        self.db = connect("127.0.0.1","root","123456","mydb" )
        self.cursor = self.db.cursor()


    def closedb():
        self.db.close()


    def select():
        sql = "SELECT * FROM student " 
        try:
            self.cursor.execute(sql)
            # 获取所有记录列表
            # results = cursor.fetchall()
            # for row in results:
            #    print(row)

            #获取前2条
            for i in range(2):
                result = self.cursor.fetchone()
                print(result)
            return result
        except:
            print ("Error: unable to fetch data")


    def insert():
        sql = """INSERT INTO EMPLOYEE(FIRST_NAME,
                LAST_NAME, AGE, SEX, INCOME)
                VALUES ('Mac', 'Mohan', 20, 'M', 2000)"""
        try:
            self.cursor.execute(sql)
            self.db.commit()
        except:
            self.db.rollback()# 如果发生错误则回滚


    def update():
        sql = "UPDATE EMPLOYEE SET AGE = AGE + 1 WHERE SEX = '%c'" % ('M')
        try:
            cursor.execute(sql)
            db.commit()
        except:
            db.rollback()

    
    def delete():
        sql = "DELETE FROM EMPLOYEE WHERE AGE > '%d'" % (20)
        try:
            cursor.execute(sql)
            db.commit()
        except:
            db.rollback()