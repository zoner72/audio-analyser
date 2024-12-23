# gui/frames/data_loader.py
from PyQt5.QtCore import QObject, pyqtSignal
import pandas as pd
import numpy as np
import librosa
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from keras.utils import to_categorical

class DataLoader(QObject):
    """
    DataLoader class for processing audio files and extracting features.
    Handles loading audio files in parallel and extracting relevant audio features.
    """
    progress_updated = pyqtSignal(int)
    
    def __init__(self, csv_path, audio_dir, n_workers=4):
        """
        Initialize the DataLoader.
        
        Args:
            csv_path (str): Path to the CSV file containing audio file information
            audio_dir (str): Directory containing the audio files
            n_workers (int): Number of parallel workers for processing
        """
        super().__init__()
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.n_workers = n_workers
        self.data = None
        self.features = []
        self.labels = []
        self.num_classes = None

    def load_metadata(self):
        """Load and validate the CSV file."""
        try:
            # Read the CSV file
            self.data = pd.read_csv(self.csv_path)
            
            # Verify required columns exist
            required_columns = ['filename', 'target']
            if not all(col in self.data.columns for col in required_columns):
                raise ValueError("CSV must contain 'filename' and 'target' columns")
            
            # Get number of unique classes
            unique_values = sorted(self.data['target'].unique())
            self.num_classes = len(unique_values)
            
            print(f"Detected {self.num_classes} unique classes")
            print(f"Class distribution:")
            for class_id in unique_values:
                count = len(self.data[self.data['target'] == class_id])
                print(f"Class {class_id}: {count} samples")
            
            print(f"Loaded metadata: {len(self.data)} entries")
            return True
            
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")

    def extract_features(self, audio_path):
        """Extract audio features from a single file."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, duration=3)
            
            # Extract features
            features = []
            
            # 1. MFCCs (increased number of coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            features.extend(mfccs_mean)
            features.extend(mfccs_std)

            # 2. Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids)
            ])

            # 3. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.extend([
                np.mean(zcr),
                np.std(zcr)
            ])

            # 4. Root Mean Square Energy
            rms = librosa.feature.rms(y=y)[0]
            features.extend([
                np.mean(rms),
                np.std(rms)
            ])

            # 5. Spectral Rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features.extend([
                np.mean(rolloff),
                np.std(rolloff)
            ])

            # 6. Chroma Features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend([
                np.mean(chroma),
                np.std(chroma)
            ])

            features = np.array(features)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

    def process_file(self, row):
        """Process a single audio file and extract its features."""
        try:
            audio_path = os.path.join(self.audio_dir, row['filename'])
            
            if not os.path.exists(audio_path):
                print(f"File not found: {audio_path}")
                return None
            
            features = self.extract_features(audio_path)
            
            if features is not None:
                # Convert label to one-hot encoding
                label = to_categorical(row['target'], num_classes=self.num_classes)
                return features, label
            
        except Exception as e:
            print(f"Error processing row: {str(e)}")
        
        return None

    def load_data(self):
        """Load and process all audio files in parallel."""
        try:
            if not self.load_metadata():
                return False
            
            total_files = len(self.data)
            processed_files = 0
            successful_files = 0
            
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                future_to_file = {
                    executor.submit(self.process_file, row): idx 
                    for idx, row in self.data.iterrows()
                }
                
                for future in as_completed(future_to_file):
                    result = future.result()
                    
                    if result is not None:
                        features, label = result
                        self.features.append(features)
                        self.labels.append(label)
                        successful_files += 1
                    
                    processed_files += 1
                    progress = int((processed_files / total_files) * 100)
                    self.progress_updated.emit(progress)
            
            if successful_files > 0:
                self.features = np.array(self.features)
                self.labels = np.array(self.labels)
                
                print(f"\nProcessing completed:")
                print(f"Total files: {total_files}")
                print(f"Successfully processed: {successful_files}")
                print(f"Features shape: {self.features.shape}")
                print(f"Labels shape: {self.labels.shape}")
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error in load_data: {str(e)}")
            return False

    def get_data(self):
        """Get the processed features and labels."""
        return self.features, self.labels, self.num_classes
