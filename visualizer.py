from PyQt6 import QtWidgets, QtGui
from PyQt6.QtGui import QPainter, QBrush, QPen, QFont
from PyQt6.QtCore import Qt



class Panel(QtWidgets.QLabel):
    def __init__(self):
        super().__init__()

        #self.message = message
        self.title = "InfoWindow"
        self.top = 100
        self.left = 100
        self.width = 700
        self.height = 500



        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.InitWindow()

    def InitWindow(self):
        self.setStyleSheet('background-color: #21bfae;')
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.setFixedWidth(self.width)
        self.setFixedHeight(self.height)
        font = QFont()
        font.setPointSize(72)
        self.setFont(font)
        self.show()

    def ChangeText(self,message):
        #all flags, checks and changes here
        self.setText(message)