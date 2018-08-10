from service import IServie



class ServiceImpl(IServie):

    def login(self):
        return True

    def d(self):
        raise Exception('子类中必须实现该方法')
    

