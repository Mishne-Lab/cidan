import numpy as np
from qtpy.QtCore import *


class MatrixSignal(QObject):
    def __init__(self):
        super().__init__()

    sig = Signal(np.ndarray)


class StrSignal(QObject):
    def __init__(self):
        super().__init__()

    sig = Signal(str)


class StrIntSignal(QObject):
    def __init__(self):
        super().__init__()

    sig = Signal(str, int)

class BoolSignal(QObject):
    def __init__(self):
        super().__init__()

    sig = Signal(bool)
