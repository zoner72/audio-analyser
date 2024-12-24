import librosa
import numpy as np

class FeatureExtractor:
    def extract_features(self, audio_data, sr):
        # Extract MFCC
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
        
        # Calculate statistics
        features = {
            'mfcc_mean': np.mean(mfccs, axis=1),
            'centroid_mean': np.mean(spectral_centroid),
            'rolloff_mean': np.mean(spectral_rolloff)
        }
        
        return features
