from PySide2.QtWidgets import *

class Module(QFrame):
    def __init__(self, importance):
        super().__init__()
        self.importance = importance