from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit

class ResultsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
    def update_results(self, text):
        self.results_text.setText(text)
