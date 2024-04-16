import sys

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np

from design import Ui_MainWindow
from image_handler import get_pixels
from neural_network import NeuralNetwork


def guess_number():
    data = get_pixels('number.png')

    nn = NeuralNetwork(trained=True)

    prediction = nn.feedforward(data)
    print(prediction)
    return np.argmax(prediction, axis=0)


class DrawingWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.canvas = QtGui.QPixmap(448, 448)
        self.canvas.fill(Qt.black)

        self.last_point = QPoint()
        self.penWidth = 32

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(13, 10, self.canvas)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.last_point = event.pos() - QPoint(self.penWidth // 2, self.penWidth // 2)
            painter = QPainter(self.canvas)
            painter.setPen(QPen(Qt.white, self.penWidth, Qt.SolidLine))
            painter.drawPoint(self.last_point)
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.last_point is not None:
            painter = QPainter(self.canvas)
            painter.setPen(QPen(Qt.white, self.penWidth, Qt.SolidLine))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos() - QPoint(self.penWidth // 2, self.penWidth // 2)
            self.update()

    def mouseReleaseEvent(self, event):
        self.last_point = None

    def save(self):
        self.update()
        scaled_canvas = self.canvas.scaled(28, 28, Qt.KeepAspectRatio)
        scaled_canvas.save('number.png')

    def clear(self):
        self.canvas = QtGui.QPixmap(448, 448)
        self.canvas.fill(Qt.black)
        self.update()


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Digit Recognition App')

        self.drawing_widget = DrawingWidget()
        self.drawing_widget.setGeometry(QtCore.QRect(26, 20, 448, 448))
        self.drawing_widget.setObjectName('canvas')

        self.canvasLayout.addWidget(self.drawing_widget)

        self.checkButton.clicked.connect(self.drawing_widget.save)
        self.checkButton.clicked.connect(self.changeAnswerText)
        self.clearButton.clicked.connect(self.drawing_widget.clear)
        self.clearButton.clicked.connect(self.clearAnswerText)

    def changeAnswerText(self):
        self.answerLabel.setText(f"I think it\'s: {guess_number()}")

    def clearAnswerText(self):
        self.answerLabel.setText("I think it\'s:")


if __name__ == '__main__':
    App = QApplication(sys.argv)

    window = Window()

    window.show()

    sys.exit(App.exec())
