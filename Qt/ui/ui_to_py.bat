:: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  --upgrade PyQt5
:: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple  --upgrade PyQt5-tools


python  -m PyQt5.uic.pyuic  Video.ui  -o  Video.py
python  -m PyQt5.uic.pyuic  set.ui  -o  set.py
python  -m PyQt5.uic.pyuic  Mainwindow.ui  -o  Mainwindow.py