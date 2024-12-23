from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit

class ModelInfoTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)
        
    def update_info(self, text):
        self.info_text.setText(text)
