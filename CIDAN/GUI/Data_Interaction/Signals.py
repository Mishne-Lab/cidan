from PySide2.QtWidgets import *
from PySide2.QtGui import *
import numpy as np
from PySide2.QtCore import *
class MatrixSignal(QObject):
    def __init__(self):
        super().__init__()

    sig = Signal(np.matrix)


class StrSignal(QObject):
    def __init__(self):
        super().__init__()

    sig = Signal(str)


class BoolSignal(QObject):
    def __init__(self):
        super().__init__()

    sig = Signal(bool)