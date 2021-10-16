from qtpy.QtCore import QObject, QEvent


class EventFilterMainWindow(QObject):
    def __init__(self, main_window):
        QObject.__init__(self)
        self.main_window = main_window

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.KeyPress):
            key = event.key()
            if self.main_window.data_handler is not None:
                current_active_main_tab_ind = self.main_window.tab_widget.currentIndex()
                current_active_main_tab = self.main_window.tabs[
                    current_active_main_tab_ind - 1]  # since dataload tab isn't in this list
                if hasattr(current_active_main_tab, "keyPressAction"):
                    return current_active_main_tab.keyPressAction(event)
        return False
