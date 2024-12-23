from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget
from .tabs.analysis_tab import AnalysisTab
from .tabs.training_tab import TrainingTab
from .tabs.results_tab import ResultsTab
from .tabs.model_info_tab import ModelInfoTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Classification System")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Add tabs
        self.analysis_tab = AnalysisTab()
        self.training_tab = TrainingTab()
        self.results_tab = ResultsTab()
        self.model_info_tab = ModelInfoTab()
        
        self.tab_widget.addTab(self.analysis_tab, "Analysis & Config")
        self.tab_widget.addTab(self.training_tab, "Model Training")
        self.tab_widget.addTab(self.results_tab, "Results")
        self.tab_widget.addTab(self.model_info_tab, "Model Info")
