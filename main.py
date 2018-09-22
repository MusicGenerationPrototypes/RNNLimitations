"""
Main
"""
import musicsource
import sys
import os
import shutil
import time
from PyQt5.QtWidgets import QWidget, QPushButton, QDesktopWidget, QApplication, QLabel
from PyQt5.QtCore import QCoreApplication, pyqtSlot, QSize
from PyQt5.QtGui import QIcon, QImage, QPalette, QBrush

class Generator(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()
        self.center()
        self.flag=True

    def initUI(self):
        gbtn = QPushButton('Сгенерировать мелодию', self)
        gbtn.clicked.connect(self.on_click) 
        gbtn.resize(gbtn.sizeHint())
        gbtn.move(150, 90)
       
        sbtn = QPushButton('Сохранить мелодию', self)
        sbtn.clicked.connect(self.open_dir) 
        sbtn.resize(gbtn.sizeHint())
        sbtn.move(150, 140)


        self.setGeometry(400, 400, 460, 250)
        initImage = QImage("background.jpg")
        bgImage = initImage.scaled(QSize(460,250))
        palette = QPalette()
        palette.setBrush(10, QBrush(bgImage))                   
        self.setPalette(palette)

        self.setWindowTitle('Music Generator')
        self.setWindowIcon(QIcon('icon.png'))
        self.show()
        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        
    def first_enter(self):
        compositor = musicsource.compositor()
        init_str = ["--test", "all", "--sample_len", "500"]
        #init_str=["--batch_size", "12"] 
        compositor.main(init_str)

    @pyqtSlot()
    def on_click(self):
        if(self.flag):
            shutil.rmtree("save\model\midi")
            self.first_enter()
            self.flag=False
        else:
            print('Файлы уже сгенерированы! Нажмите на кнопку \"Сохранить мелодию\" для продолжения работы' )

    @pyqtSlot()
    def open_dir(self):
        shutil.move("save\model\midi", "save\output")
        os.rename("save\output\midi", "save\output\midi" + time.strftime("%Y%m%d%H%M%S"))
        os.makedirs("save\model\midi", exist_ok=True)
        os.startfile("save\output")
        sys.exit()
        
if __name__ == "__main__":  
    app = QApplication(sys.argv)
    ex = Generator()   
    sys.exit(app.exec_())