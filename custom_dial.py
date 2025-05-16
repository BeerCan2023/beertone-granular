from PyQt5.QtWidgets import QDial
from PyQt5.QtCore import Qt, QPoint

class CustomDial(QDial):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_y = None
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._last_y = event.globalY()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._last_y is not None and event.buttons() & Qt.LeftButton:
            dy = self._last_y - event.globalY()  # Sens intuitif : haut augmente, bas diminue
            step = self.singleStep() if hasattr(self, 'singleStep') else 1
            new_value = self.value() + dy * step
            self.setValue(min(max(self.minimum(), new_value), self.maximum()))
            self._last_y = event.globalY()
            # NE PAS appeler super() ici : on bloque le comportement natif horizontal
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._last_y is not None:
            # On a fait un drag vertical, on bloque le mouseRelease natif
            self._last_y = None
            event.accept()
            return
        super().mouseReleaseEvent(event)
