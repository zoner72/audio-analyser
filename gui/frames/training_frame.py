from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QLabel, QFileDialog, QProgressBar, QMessageBox,
                            QTextEdit, QScrollArea)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import os
import json
from datetime import datetime
from .data_loader import DataLoader
from keras.callbacks import Callback
from keras.regularizers import l2

class DataLoadWorker(QThread):
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(object, object, int)
    error = pyqtSignal(str)

    def __init__(self, csv_path, audio_dir):
        super().__init__()
        self.csv_path = csv_path
        self.audio_dir = audio_dir

    def run(self):
        try:
            data_loader = DataLoader(self.csv_path, self.audio_dir)
            data_loader.progress_updated.connect(self.progress_updated.emit)
            
            if data_loader.load_data():
                features, labels, num_classes = data_loader.get_data()
                self.finished.emit(features, labels, num_classes)
            else:
                self.error.emit("Failed to load data")
        except Exception as e:
            self.error.emit(str(e))

class CustomTrainingCallback(Callback):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker
        self.best_val_accuracy = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # Track best validation accuracy
            current_val_accuracy = logs.get('val_accuracy', 0)
            if current_val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = current_val_accuracy
                self.best_epoch = epoch + 1

            # Add best metrics to logs
            logs['best_val_accuracy'] = self.best_val_accuracy
            logs['best_epoch'] = self.best_epoch

        self.worker.epoch_completed.emit(logs)
        progress = int((epoch + 1) / self.params['epochs'] * 100)
        self.worker.progress_updated.emit(progress)

class TrainingWorker(QThread):
    progress_updated = pyqtSignal(int)
    epoch_completed = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, features, labels, num_classes):
        super().__init__()
        self.features = features
        self.labels = labels
        self.num_classes = num_classes
        self.model = None

    def run(self):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, BatchNormalization
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
            from keras.optimizers import Adam

            # Preprocess the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.features)
            
            # Split the data
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, self.labels, test_size=0.2, random_state=42
            )
            
            # Create improved model architecture
            self.model = Sequential([
                # Input layer with stronger regularization
                Dense(128, activation='relu', 
                      input_shape=(X_train.shape[1],),
                      kernel_regularizer=l2(0.01)),
                BatchNormalization(),
                Dropout(0.4),
                
                # Hidden layer 1
                Dense(96, activation='relu',
                      kernel_regularizer=l2(0.01)),
                BatchNormalization(),
                Dropout(0.4),
                
                # Hidden layer 2
                Dense(64, activation='relu',
                      kernel_regularizer=l2(0.01)),
                BatchNormalization(),
                Dropout(0.3),
                
                # Output layer
                Dense(self.num_classes, activation='softmax',
                      kernel_regularizer=l2(0.01))
            ])
            
            # Compile with reduced learning rate
            optimizer = Adam(learning_rate=0.0005)
            self.model.compile(
                optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Enhanced callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
            
            custom_callback = CustomTrainingCallback(self)
            
            # Train with smaller batch size
            history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=16,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr, custom_callback],
                verbose=0
            )
            
            # Evaluate the model
            test_loss, test_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
            
            # Prepare comprehensive results
            results = {
                'model': self.model,
                'history': history.history,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'best_val_accuracy': custom_callback.best_val_accuracy,
                'best_epoch': custom_callback.best_epoch,
                'model_summary': [
                    {
                        'name': layer.__class__.__name__,
                        'config': {
                            k: str(v) if isinstance(v, (dict, list)) else v
                            for k, v in layer.get_config().items()
                        },
                        'params': layer.count_params()
                    }
                    for layer in self.model.layers
                ],
                'training_params': {
                    'batch_size': 16,
                    'initial_learning_rate': 0.0005,
                    'l2_regularization': 0.01,
                    'dropout_rate': 0.4
                }
            }
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))
            
class TrainingFrame(QFrame):
    def __init__(self):
        super().__init__()
        self.data_loader = None
        self.worker = None
        self.training_worker = None
        self.features = None
        self.labels = None
        self.num_classes = None
        self.model = None
        self.model_info = {}
        self.training_history = {}
        self.setup_ui()

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        
        # CSV selection section
        csv_section = QFrame()
        csv_layout = QHBoxLayout(csv_section)
        
        self.csv_label = QLabel("CSV File:")
        self.csv_path_label = QLabel("No file selected")
        self.csv_button = QPushButton("Select CSV")
        self.csv_button.clicked.connect(self.select_csv)
        
        csv_layout.addWidget(self.csv_label)
        csv_layout.addWidget(self.csv_path_label)
        csv_layout.addWidget(self.csv_button)
        
        # Audio directory selection section
        audio_dir_section = QFrame()
        audio_dir_layout = QHBoxLayout(audio_dir_section)
        
        self.audio_dir_label = QLabel("Audio Directory:")
        self.audio_dir_path_label = QLabel("No directory selected")
        self.audio_dir_button = QPushButton("Select Directory")
        self.audio_dir_button.clicked.connect(self.select_audio_dir)
        
        audio_dir_layout.addWidget(self.audio_dir_label)
        audio_dir_layout.addWidget(self.audio_dir_path_label)
        audio_dir_layout.addWidget(self.audio_dir_button)
        
        # Status and progress section
        status_section = QFrame()
        status_layout = QVBoxLayout(status_section)
        
        # Status Label
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        
        # Data loading progress
        loading_progress_section = QFrame()
        loading_progress_layout = QHBoxLayout(loading_progress_section)
        self.loading_label = QLabel("Data Loading Progress:")
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 100)
        loading_progress_layout.addWidget(self.loading_label)
        loading_progress_layout.addWidget(self.loading_progress)
        
        # Training progress
        training_progress_section = QFrame()
        training_progress_layout = QHBoxLayout(training_progress_section)
        self.training_label = QLabel("Training Progress:")
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        training_progress_layout.addWidget(self.training_label)
        training_progress_layout.addWidget(self.training_progress)
        
        # Training log with scroll area
        log_section = QFrame()
        log_layout = QVBoxLayout(log_section)
        self.log_label = QLabel("Training Log:")
        
        # Create scroll area for log
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(200)
        
        # Create widget to hold log text
        log_container = QFrame()
        log_container_layout = QVBoxLayout(log_container)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        log_container_layout.addWidget(self.log_text)
        
        scroll_area.setWidget(log_container)
        log_layout.addWidget(self.log_label)
        log_layout.addWidget(scroll_area)
        
        status_layout.addWidget(loading_progress_section)
        status_layout.addWidget(training_progress_section)
        status_layout.addWidget(log_section)
        
        # Button section
        button_section = QFrame()
        button_layout = QHBoxLayout(button_section)
        
        self.load_button = QPushButton("Load Training Data")
        self.load_button.clicked.connect(self.load_training_data)
        
        self.train_button = QPushButton("Start Training")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False)
        
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.train_button)

        # Model Information Display
        info_section = QFrame()
        info_layout = QVBoxLayout(info_section)
        
        self.info_label = QLabel("Model Information:")
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(150)
        
        info_layout.addWidget(self.info_label)
        info_layout.addWidget(self.info_text)

        # Save Model Button
        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        
        # Add all sections to main layout
        self.main_layout.addWidget(csv_section)
        self.main_layout.addWidget(audio_dir_section)
        self.main_layout.addWidget(status_section)
        self.main_layout.addWidget(button_section)
        self.main_layout.addWidget(info_section)
        self.main_layout.addWidget(self.save_button)
        self.main_layout.addStretch()

    def select_csv(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv)"
        )
        if file_name:
            self.csv_path_label.setText(file_name)

    def select_audio_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Audio Directory"
        )
        if directory:
            self.audio_dir_path_label.setText(directory)

    def load_training_data(self):
        csv_path = self.csv_path_label.text()
        audio_dir = self.audio_dir_path_label.text()
        
        if csv_path == "No file selected" or audio_dir == "No directory selected":
            QMessageBox.warning(self, "Error", "Please select both CSV file and audio directory")
            return
        
        self.load_button.setEnabled(False)
        self.train_button.setEnabled(False)
        self.loading_progress.setValue(0)
        self.status_label.setText("Loading data...")
        self.log_text.clear()
        
        self.worker = DataLoadWorker(csv_path, audio_dir)
        self.worker.progress_updated.connect(self.update_loading_progress)
        self.worker.finished.connect(self.on_data_loaded)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def update_loading_progress(self, value):
        self.loading_progress.setValue(value)

    def update_training_progress(self, value):
        self.training_progress.setValue(value)

    def update_training_log(self, logs):
        if logs:
            current_epoch = len(self.training_history.get('loss', [])) + 1
            log_text = f"\nEpoch {current_epoch}:\n"
            
            # Format metrics
            metrics = {
                'loss': ('Loss', logs.get('loss', 0)),
                'accuracy': ('Accuracy', logs.get('accuracy', 0)),
                'val_loss': ('Val Loss', logs.get('val_loss', 0)),
                'val_accuracy': ('Val Accuracy', logs.get('val_accuracy', 0)),
                'best_val_accuracy': ('Best Val Accuracy', logs.get('best_val_accuracy', 0)),
                'best_epoch': ('Best Epoch', logs.get('best_epoch', 0))
            }
            
            # Add formatted metrics to log
            for metric_name, (display_name, value) in metrics.items():
                if metric_name in logs:
                    if isinstance(value, float):
                        log_text += f"  {display_name}: {value:.4f}\n"
                    else:
                        log_text += f"  {display_name}: {value}\n"
            
            self.log_text.append(log_text)
            
            # Scroll to bottom
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
            # Store history
            for key, value in logs.items():
                self.training_history.setdefault(key, []).append(value)

    def on_data_loaded(self, features, labels, num_classes):
        self.features = features
        self.labels = labels
        self.num_classes = num_classes
        self.log_text.append(f"Loaded {len(features)} samples successfully\n")
        self.status_label.setText("Data loaded successfully")
        self.load_button.setEnabled(True)
        self.train_button.setEnabled(True)
        self.worker = None

    def on_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.status_label.setText("Error occurred")
        self.log_text.append(f"Error: {error_message}\n")
        self.load_button.setEnabled(True)
        self.worker = None

    def start_training(self):
        if self.features is None or self.labels is None:
            QMessageBox.warning(self, "Error", "Please load training data first")
            return
        
        self.status_label.setText("Training in progress...")
        self.train_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.training_progress.setValue(0)
        self.log_text.append("\nTraining started...\n")
        self.training_history.clear()
        
        self.training_worker = TrainingWorker(self.features, self.labels, self.num_classes)
        self.training_worker.progress_updated.connect(self.update_training_progress)
        self.training_worker.epoch_completed.connect(self.update_training_log)
        self.training_worker.finished.connect(self.on_training_finished)
        self.training_worker.error.connect(self.on_training_error)
        self.training_worker.start()

    def on_training_finished(self, results):
        self.model = results['model']
        
        # Update model info with comprehensive results
        self.model_info = {
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'architecture': {
                'input_shape': self.features.shape[1],
                'num_classes': self.num_classes,
                'layers': [
                    f"{layer['name']} - Parameters: {layer['params']}"
                    for layer in results['model_summary']
                ]
            },
            'training_params': results['training_params'],
            'performance': {
                'final_loss': float(results['test_loss']),
                'final_accuracy': float(results['test_accuracy']),
                'best_val_accuracy': float(results['best_val_accuracy']),
                'best_epoch': int(results['best_epoch'])
            },
            'dataset_info': {
                'total_samples': len(self.features),
                'training_samples': int(len(self.features) * 0.8),
                'validation_samples': int(len(self.features) * 0.2)
            }
        }
        
        self.update_model_info()
        final_message = (
            f"\nTraining completed!\n"
            f"Final Accuracy: {results['test_accuracy']:.2%}\n"
            f"Best Validation Accuracy: {results['best_val_accuracy']:.2%} "
            f"(Epoch {results['best_epoch']})\n"
        )
        self.log_text.append(final_message)
        self.status_label.setText(f"Training completed - Accuracy: {results['test_accuracy']:.2%}")
        self.train_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.training_worker = None

    def on_training_error(self, error_message):
        QMessageBox.critical(self, "Error", f"Training error: {error_message}")
        self.status_label.setText("Training failed!")
        self.log_text.append(f"\nTraining failed: {error_message}\n")
        self.train_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.training_worker = None

    def update_model_info(self):
        info_text = "Model Information:\n\n"
        
        # Helper function for formatting nested dictionaries
        def format_dict(d, indent=0):
            text = ""
            for key, value in d.items():
                if isinstance(value, dict):
                    text += f"{'  ' * indent}{key}:\n"
                    text += format_dict(value, indent + 1)
                elif isinstance(value, list):
                    text += f"{'  ' * indent}{key}:\n"
                    for item in value:
                        if isinstance(item, dict):
                            text += format_dict(item, indent + 1)
                        else:
                            text += f"{'  ' * (indent + 1)}{item}\n"
                elif isinstance(value, float):
                    text += f"{'  ' * indent}{key}: {value:.4f}\n"
                else:
                    text += f"{'  ' * indent}{key}: {value}\n"
            return text
        
        info_text += format_dict(self.model_info)
        self.info_text.setText(info_text)

    def save_model(self):
        try:
            save_dir = QFileDialog.getExistingDirectory(
                self, "Select Directory to Save Model"
            )
            
            if save_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_dir = os.path.join(save_dir, f"model_{timestamp}")
                os.makedirs(model_dir, exist_ok=True)
                
                # Save model
                model_path = os.path.join(model_dir, "model.h5")
                self.model.save(model_path)
                
                # Save model information
                info_path = os.path.join(model_dir, "model_info.json")
                with open(info_path, 'w') as f:
                    json.dump(self.model_info, f, indent=4)
                
                # Save training history
                history_path = os.path.join(model_dir, "training_history.json")
                with open(history_path, 'w') as f:
                    json.dump(self.training_history, f, indent=4)
                
                QMessageBox.information(
                    self, 
                    "Success", 
                    f"Model and information saved to:\n{model_dir}"
                )
                
                self.log_text.append(f"\nModel saved to: {model_dir}\n")
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to save model: {str(e)}"
            )
            self.log_text.append(f"\nError saving model: {str(e)}\n")

