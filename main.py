import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from design import Ui_MainWindow


class DrawingWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.lastPoint = QPoint()

        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)

        self.painter = QPainter()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            self.painter.begin(self.image)
            self.painter.setPen(QPen(Qt.white, 8.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            self.painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()
            self.painter.end()
            print(2)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def saveImage(self, path):
        pixmap = self.grab()
        pixmap.save(path)

    def paintEvent(self, event):
        canvasPainter = QPainter(self)

        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        print(1)

    def clear(self):
        # make the whole canvas white
        self.image.fill(Qt.black)
        # update
        self.update()


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.drawing_widget = DrawingWidget(self.mainLayout)
        self.drawing_widget.setGeometry(QtCore.QRect(26, 20, 448, 448))
        self.drawing_widget.setObjectName("canvas")


if __name__ == '__main__':
    App = QApplication(sys.argv)

    window = Window()

    window.show()

    sys.exit(App.exec())
